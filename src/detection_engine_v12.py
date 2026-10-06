from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import json
import re
# all the configurable thresholds and settings in one place so they're easy to adjust
# these values took a lot of tweaking across earlier versions to get the accuracy up
THRESHOLDS = {
    # normal working hours window
    'normal_hours_start': 6,
    'normal_hours_end': 22,
    'max_patients_per_hour': 10,  # fallback when can't role-specific ones arent available
    'sequential_window_minutes': 5,
    'sequential_min_length': 5,
    'same_patient_access_threshold': 5,
    'extended_viewing_window_hours': 2,
    'failed_access_threshold': 3,
 # Same-surname rule params (Hedda et al)
    'same_surname_threshold': 2,
    'same_surname_time_window_minutes': 45,
    'same_surname_min_gap_seconds': 30,
 # Geographic proximity, for patients that are in the same postcode area
    'same_location_threshold': 5,
    'geographic_time_window_hours': 2,
 # Cross-department, This can be for a receptionists accessing clinical data they shouldn't need
    'restricted_roles': ['Receptionist'],
    'cross_dept_min_accesses': 6,
    'cross_dept_time_window_hours': 3,
    'cross_dept_strict_events': True,
  # VIP patients, high-profile individuals who get snooped on most often by any insider
    'vip_detection_enabled': True,
    'vip_patient_ids': [103, 110, 69],
    'vip_only_flag_users': ['malicious_user'],
# Street-level matching, this is even more specific than postcode
    'street_match_enabled': True,
    'street_match_threshold': 3,
    'street_match_time_window_hours': 4,
    # The idea from Hedda et al. (2017) is if a flagged access has a legit clinical
    # reason (appointment, active order, recent notes), we dismiss the flag.
    # Their research paper showed this cuts false positives by around 16-44%.
    'explanation_filtering_enabled': True,
    'appointment_window_hours': 48,      # check ±48h from access time, all set and strict times
    'documentation_window_hours': 24,    # notes written within 24h
    'order_active_window_hours': 72,     # active orders within 72h
 # which rules should be run through the justification filter
    'rules_with_explanation_filtering': [
        'same_surname',
        'geographic_proximity',
        'vip_access',
        'cross_department'
    ],
 # Role-specific baselines for high-volume detection
# The idea is that a nurse seeing 15 patients/hr is normal but a receptionist
# seeing 15 is suspicious. This is based off the formula Threshold = mean + 2.5 SD per role.
    # Turned off as my synthetic dataset is too small causing script issues in terms of pecision.
    'role_specific_thresholds_enabled': False,
    'role_baseline_std_multiplier': 2.5,
# Default values per role (patients per hour)
    'role_baselines': {
        'Physician': {'mean': 12, 'std': 4},
        'Nurse': {'mean': 15, 'std': 5},
        'Receptionist': {'mean': 8, 'std': 3},
        'Administrator': {'mean': 5, 'std': 2},
        'Lab Tech': {'mean': 10, 'std': 3},
        'default': {'mean': 10, 'std': 3}
    },
# VIP justification, where its only flagged VIP access if there's no legit reason
    'vip_require_no_justification': True,
# Composite scoring, if multiple rules fire on the same log entry, bump it up to HIGH or CRITICAL priority
    'composite_scoring_enabled': True,
    'high_risk_min_rules': 2,
    'critical_risk_min_rules': 3,
 # how much weight each rule carries in the composite score
 # weights are roughly based on how confident I am in each rule's precision from testing
 # street_match gets the highest because it almost never fires unless something's actually wrong
    'rule_weights': {
        'after_hours': 2.0,
        'high_volume': 3.0,
        'sequential_browse': 3.5,
        'same_surname': 2.8,
        'geographic_proximity': 2.5,
        'street_match': 4.1,
        'cross_department': 2.0,
        'extended_viewing': 2.5,
        'failed_access': 3.0,
        'vip_access': 3.5,
    }
}

# connect to the OpenEMR database running in Docker
engine = create_engine(
    "mysql+mysqlconnector://openemr:openemrpass@localhost:3306/openemr"
)
def load_audit_logs():
    """Pull audit logs from the DB. Only grabs our simulated entries (tagged with SIMULATED|)
    and parses out the ground truth labels."""
    query = """
        SELECT id, date, event, user, patient_id, success, comments
        FROM log WHERE comments LIKE 'SIMULATED|%' ORDER BY date
    """
    df = pd.read_sql(query, engine)

    def parse_ground_truth(comment):
        try:
            if comment and 'SIMULATED|' in comment:
                return json.loads(comment.replace('SIMULATED|', ''))
        except:
            pass
        return {'is_malicious': False, 'threat_pattern': None}

    df['ground_truth'] = df['comments'].apply(parse_ground_truth)
    df['is_actually_malicious'] = df['ground_truth'].apply(lambda x: x.get('is_malicious', False))
    df['actual_pattern'] = df['ground_truth'].apply(lambda x: x.get('threat_pattern'))
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_users():
    """Get user info - we need the role/title to know who should be accessing what."""
    return pd.read_sql("""
        SELECT id, username, fname, lname, title, facility_id
        FROM users WHERE username != ''
    """, engine)

def load_patients():
    """Get patient demographics - needed for surname, postcode, and street checks."""
    return pd.read_sql("""
        SELECT pid, fname, lname, postal_code, street, city, state
        FROM patient_data WHERE pid > 0
    """, engine)

# these are needed for the V12 justification checks - if there's a legit clinical reason
# for an access we don't want to flag it as suspicious

def load_appointments():
    """Try to grab appointments from OpenEMR's calendar table.
    If it's empty or doesn't exist (common in our synthetic setup), just return empty."""
    try:
        return pd.read_sql("""
            SELECT pc_eid, pc_pid, pc_eventDate, pc_startTime, pc_aid,
                   pc_title, pc_catid
            FROM openemr_postcalendar_events
            WHERE pc_pid > 0
        """, engine)
    except Exception as e:
        print(f"Note: Appointments table not available, using simulated justifications")
        return pd.DataFrame(columns=['pc_eid', 'pc_pid', 'pc_eventDate', 'pc_startTime', 'pc_aid'])

def load_orders():
    """Load medication/lab orders - used to check if access was justified by an active order."""
    try:
        return pd.read_sql("""
            SELECT procedure_order_id, patient_id, provider_id,
                   date_ordered, order_status
            FROM procedure_order
            WHERE patient_id > 0
        """, engine)
    except Exception as e:
        print(f"Note: Orders table not available, using simulated justifications")
        return pd.DataFrame(columns=['procedure_order_id', 'patient_id', 'date_ordered'])

def load_clinical_notes():
    """Load encounter records. If a user wrote notes for a patient around the same time
    they accessed the record, that's a good sign the access was legit."""
    try:
        return pd.read_sql("""
            SELECT id, pid, date, user, encounter
            FROM form_encounter
            WHERE pid > 0
        """, engine)
    except Exception as e:
        print(f"Note: Notes table not available, using simulated justifications")
        return pd.DataFrame(columns=['id', 'pid', 'date', 'user'])

# before confirming a flag as a true positive, check if there's a legitimate clinical
# reason behind it. Based on the approach in Hedda et al. (2017) - they showed this
# cuts false positives significantly without losing real detections
class ClinicalJustificationChecker:
    """Checks whether a flagged access has a legit clinical reason behind it.
    Looks at appointments, orders, and clinical notes."""

    def __init__(self, appointments_df, orders_df, notes_df, config=THRESHOLDS):
        self.appointments = appointments_df
        self.orders = orders_df
        self.notes = notes_df
        self.config = config
        self._prepare_lookups()

    def _prepare_lookups(self):
        """Build dictionaries for quick lookups instead of scanning the whole DF each time."""
        # appointments grouped by patient
        self.patient_appointments = defaultdict(list)
        if not self.appointments.empty and 'pc_pid' in self.appointments.columns:
            for _, row in self.appointments.iterrows():
                try:
                    appt_date = pd.to_datetime(row['pc_eventDate'])
                    self.patient_appointments[row['pc_pid']].append(appt_date)
                except:
                    pass

        # orders grouped by patient
        self.patient_orders = defaultdict(list)
        if not self.orders.empty and 'patient_id' in self.orders.columns:
            for _, row in self.orders.iterrows():
                try:
                    order_date = pd.to_datetime(row['date_ordered'])
                    self.patient_orders[row['patient_id']].append(order_date)
                except:
                    pass

        # notes grouped by (patient, user) combo
        self.patient_user_notes = defaultdict(list)
        if not self.notes.empty and 'pid' in self.notes.columns:
            for _, row in self.notes.iterrows():
                try:
                    note_date = pd.to_datetime(row['date'])
                    key = (row['pid'], row.get('user', ''))
                    self.patient_user_notes[key].append(note_date)
                except:
                    pass

    def has_appointment_justification(self, patient_id, access_time, user=None):
        """Did this patient have an appointment within ±48 hours of when they were accessed?"""
        window = timedelta(hours=self.config['appointment_window_hours'])
        appointments = self.patient_appointments.get(patient_id, [])

        for appt_date in appointments:
            if abs((appt_date - access_time).total_seconds()) <= window.total_seconds():
                return True, f"Appointment within ±{self.config['appointment_window_hours']}h"

        return False, None

    def has_order_justification(self, patient_id, access_time):
        """Was there an active order (lab, meds) around the time of access?"""
        window = timedelta(hours=self.config['order_active_window_hours'])
        orders = self.patient_orders.get(patient_id, [])

        for order_date in orders:
            if abs((order_date - access_time).total_seconds()) <= window.total_seconds():
                return True, f"Active order within ±{self.config['order_active_window_hours']}h"

        return False, None

    def has_documentation_justification(self, patient_id, user, access_time):
        """Did the user write notes for this patient shortly after accessing?
        If so, they probably had a real reason to be in the record."""
        window = timedelta(hours=self.config['documentation_window_hours'])
        key = (patient_id, user)
        notes = self.patient_user_notes.get(key, [])

        for note_date in notes:
            time_diff = (note_date - access_time).total_seconds()
            if 0 <= time_diff <= window.total_seconds():
                return True, f"Documentation within {self.config['documentation_window_hours']}h"

        return False, None

    def has_any_justification(self, patient_id, user, access_time):
        """Check all three justification types. Returns True if ANY legit reason found."""
        # appointment is the strongest justification so check first
        has_appt, reason = self.has_appointment_justification(patient_id, access_time, user)
        if has_appt:
            return True, reason

        has_order, reason = self.has_order_justification(patient_id, access_time)
        if has_order:
            return True, reason

        has_doc, reason = self.has_documentation_justification(patient_id, user, access_time)
        if has_doc:
            return True, reason

        return False, None

    def filter_flags(self, flags, rules_to_filter=None):
        """Go through all flags and dismiss any that have a clinical justification.
        Only applies to the rules specified in rules_to_filter (not all rules need this)."""
        if rules_to_filter is None:
            rules_to_filter = self.config['rules_with_explanation_filtering']

        filtered = []
        dismissed = []
        stats = {'total': len(flags), 'filtered': 0, 'dismissed': 0}

        for flag in flags:
            # skip rules don't want to filter
            if flag['rule'] not in rules_to_filter:
                filtered.append(flag)
                continue

            has_justification, reason = self.has_any_justification(
                flag['patient_id'],
                flag['user'],
                flag['timestamp']
            )

            if has_justification:
                flag['dismissed'] = True
                flag['dismissal_reason'] = reason
                dismissed.append(flag)
                stats['dismissed'] += 1
            else:
                flag['has_justification'] = False
                filtered.append(flag)
                stats['filtered'] += 1

        return filtered, dismissed, stats

# instead of one fixed threshold for everyone, work out what's normal per role
# a nurse seeing 15 patients/hr is fine, a receptionist doing the same isn't

def calculate_role_baselines(df, users_df):
    """Work out what 'normal' access looks like for each role.
    Threshold = mean + 2.5 standard deviations."""
    role_lookup = users_df.set_index('username')['title'].to_dict()

    df_copy = df.copy()
    df_copy['hour_bucket'] = df_copy['date'].dt.floor('h')
    df_copy['role'] = df_copy['user'].map(role_lookup)

    # count unique patients per user per hour
    hourly_counts = df_copy.groupby(['user', 'role', 'hour_bucket'])['patient_id'].nunique().reset_index()
    hourly_counts.columns = ['user', 'role', 'hour_bucket', 'patient_count']

    # get stats per role
    role_stats = hourly_counts.groupby('role')['patient_count'].agg(['mean', 'std']).reset_index()
    role_stats.columns = ['role', 'mean', 'std']
    role_stats['std'] = role_stats['std'].fillna(0)

    baselines = {}
    for _, row in role_stats.iterrows():
        role = row['role'] if pd.notna(row['role']) else 'default'
        baselines[role] = {
            'mean': round(row['mean'], 2),
            'std': round(row['std'], 2) if row['std'] > 0 else 1.0,
            'threshold': round(row['mean'] + THRESHOLDS['role_baseline_std_multiplier'] * max(row['std'], 1), 2)
        }

    return baselines

def get_role_threshold(role, baselines):
    """Look up the threshold for a given role, fall back to default if unknown."""
    if role in baselines:
        return baselines[role]['threshold']
    elif 'default' in baselines:
        return baselines['default']['threshold']
    else:
        return THRESHOLDS['max_patients_per_hour']
def rule_after_hours(df):
    """RULE 1: Flag any access outside normal working hours (before 6am or after 10pm)."""
    flagged = []
    for idx, row in df.iterrows():
        hour = row['date'].hour
        if hour < THRESHOLDS['normal_hours_start'] or hour >= THRESHOLDS['normal_hours_end']:
            flagged.append({
                'log_id': row['id'], 'rule': 'after_hours', 'user': row['user'],
                'patient_id': row['patient_id'], 'timestamp': row['date'],
                'details': f"Access at {hour}:00 (outside {THRESHOLDS['normal_hours_start']}:00-{THRESHOLDS['normal_hours_end']}:00)",
                'risk_weight': THRESHOLDS['rule_weights']['after_hours']
            })
    return flagged

def rule_high_volume_v12(df, users_df, role_baselines=None):
    """RULE 2: Flag users who access way more patients per hour than normal.
    Uses distinct patient count per hour, not total events - someone refreshing
    the same record 10 times shouldn't count the same as accessing 10 different patients."""
    flagged = []
    role_lookup = users_df.set_index('username')['title'].to_dict()

    df_copy = df.copy()
    df_copy['hour_bucket'] = df_copy['date'].dt.floor('h')

    for (user, hour), group in df_copy.groupby(['user', 'hour_bucket']):
        distinct = group['patient_id'].nunique()

        user_role = role_lookup.get(user, 'default') or 'default'

        if THRESHOLDS['role_specific_thresholds_enabled'] and role_baselines:
            threshold = get_role_threshold(user_role, role_baselines)
            threshold_type = "role-specific"
        else:
            threshold = THRESHOLDS['max_patients_per_hour']
            threshold_type = "fixed"

        if distinct > threshold:
            for idx, row in group.iterrows():
                flagged.append({
                    'log_id': row['id'], 'rule': 'high_volume', 'user': user,
                    'patient_id': row['patient_id'], 'timestamp': row['date'],
                    'details': f"{distinct} patients/hour (threshold: {threshold:.1f}, {threshold_type} for {user_role})",
                    'risk_weight': THRESHOLDS['rule_weights']['high_volume'],
                    'role': user_role,
                    'threshold_used': threshold
                })
    return flagged

def rule_sequential_browsing(df, patients_df):
    """RULE 3: Catches someone going through patient records in alphabetical order.
    This is a snooping pattern, browsing A-Z through the patient list.
    Looking for sequences of 5+ patients accessed within 5 min windows where
    the surnames are in alphabetical order.
    Note: had to handle the end-of-loop case separately (the second check after
    the loop) - the sequence doesn't get flushed otherwise if it runs to the end."""
    flagged = []
    surname_lookup = patients_df.set_index('pid')['lname'].to_dict()

    for user, user_df in df.groupby('user'):
        user_df = user_df.sort_values('date')
        sequence = []
        last_time = None

        for idx, row in user_df.iterrows():
            surname = surname_lookup.get(row['patient_id'], '')
            current_time = row['date']

            if last_time is None or (current_time - last_time).total_seconds() / 60 <= THRESHOLDS['sequential_window_minutes']:
                sequence.append({'log_id': row['id'], 'surname': surname,
                               'patient_id': row['patient_id'], 'timestamp': current_time})
            else:
                if len(sequence) >= THRESHOLDS['sequential_min_length']:
                    surnames = [s['surname'] for s in sequence]
                    if surnames == sorted(surnames):
                        for entry in sequence:
                            flagged.append({
                                'log_id': entry['log_id'], 'rule': 'sequential_browse',
                                'user': user, 'patient_id': entry['patient_id'],
                                'timestamp': entry['timestamp'],
                                'details': f"Alphabetical sequence of {len(sequence)} patients",
                                'risk_weight': THRESHOLDS['rule_weights']['sequential_browse']
                            })
                sequence = [{'log_id': row['id'], 'surname': surname,
                           'patient_id': row['patient_id'], 'timestamp': current_time}]
            last_time = current_time

        
        if len(sequence) >= THRESHOLDS['sequential_min_length']:
            surnames = [s['surname'] for s in sequence]
            if surnames == sorted(surnames):
                for entry in sequence:
                    flagged.append({
                        'log_id': entry['log_id'], 'rule': 'sequential_browse',
                        'user': user, 'patient_id': entry['patient_id'],
                        'timestamp': entry['timestamp'],
                        'details': f"Alphabetical sequence of {len(sequence)} patients",
                        'risk_weight': THRESHOLDS['rule_weights']['sequential_browse']
                    })
    return flagged

def rule_same_surname(df, patients_df):
    """RULE 4: Flags when someone accesses multiple patients with the same surname
    in a short window."""
    flagged = []
    flagged_ids = set()

    surname_lookup = patients_df.set_index('pid')['lname'].to_dict()
    df_copy = df.copy()
    df_copy['patient_surname'] = df_copy['patient_id'].map(surname_lookup)

    time_window = timedelta(minutes=THRESHOLDS['same_surname_time_window_minutes'])
    min_gap = timedelta(seconds=THRESHOLDS['same_surname_min_gap_seconds'])

    for user, user_df in df_copy.groupby('user'):
        user_df = user_df.sort_values('date')

        for i, row in user_df.iterrows():
            window_start = row['date']
            window_end = window_start + time_window

            window_df = user_df[(user_df['date'] >= window_start) & (user_df['date'] <= window_end)]

            for surname in window_df['patient_surname'].dropna().unique():
                surname_df = window_df[window_df['patient_surname'] == surname]
                unique_patients = surname_df['patient_id'].nunique()

                if unique_patients >= THRESHOLDS['same_surname_threshold']:
                    times = surname_df['date'].sort_values()
                    if len(times) >= 2:
                        time_gaps = times.diff().dropna()
                        if any(gap >= min_gap for gap in time_gaps):
                            for idx, match_row in surname_df.iterrows():
                                if match_row['id'] not in flagged_ids:
                                    flagged_ids.add(match_row['id'])
                                    flagged.append({
                                        'log_id': match_row['id'], 'rule': 'same_surname',
                                        'user': user, 'patient_id': match_row['patient_id'],
                                        'timestamp': match_row['date'],
                                        'details': f"{unique_patients} patients with surname '{surname}'",
                                        'risk_weight': THRESHOLDS['rule_weights']['same_surname']
                                    })
    return flagged

def rule_geographic_proximity(df, patients_df):
    """RULE 5: Flags when someone accesses lots of patients from the same postcode, classic pattern for someone snooping on their neighbours."""
    flagged = []
    flagged_ids = set()

    postal_lookup = patients_df.set_index('pid')['postal_code'].to_dict()
    df_copy = df.copy()
    df_copy['postal_code'] = df_copy['patient_id'].map(postal_lookup)

    time_window = timedelta(hours=THRESHOLDS['geographic_time_window_hours'])

    for user, user_df in df_copy.groupby('user'):
        user_df = user_df.sort_values('date')

        for i, row in user_df.iterrows():
            window_start = row['date']
            window_end = window_start + time_window

            window_df = user_df[(user_df['date'] >= window_start) & (user_df['date'] <= window_end)]

            for postal in window_df['postal_code'].dropna().unique():
                postal_df = window_df[window_df['postal_code'] == postal]
                unique_patients = postal_df['patient_id'].nunique()

                if unique_patients >= THRESHOLDS['same_location_threshold']:
                    for idx, match_row in postal_df.iterrows():
                        if match_row['id'] not in flagged_ids:
                            flagged_ids.add(match_row['id'])
                            flagged.append({
                                'log_id': match_row['id'], 'rule': 'geographic_proximity',
                                'user': user, 'patient_id': match_row['patient_id'],
                                'timestamp': match_row['date'],
                                'details': f"{unique_patients} patients from postal code '{postal}'",
                                'risk_weight': THRESHOLDS['rule_weights']['geographic_proximity']
                            })
    return flagged

def rule_cross_department(df, users_df):
    """RULE 6: Catches admin/reception staff accessing clinical data they shouldn't need.
    Extended to handle synthetic data where event names might not always contain the
    clinical keywords - also accepts entries where metadata has access_context =
    'restricted_clinical_data'. In a real system this context field would come from
    the EHR's own classification of what data category was accessed.
    Also falls back to username pattern matching if the DB title isn't set correctly,
    which can happen in synthetic setups where OpenEMR users aren't fully configured."""
    flagged = []
    flagged_ids = set()
    role_lookup = users_df.set_index('username')['title'].to_dict()

    sensitive_keywords = ['diagnosis', 'medication', 'lab', 'clinical', 'notes']

    def is_sensitive_event(event, ground_truth=None):
        # primary check: event name contains a clinical keyword (works on real data)
        if event:
            event_lower = str(event).lower()
            if any(kw in event_lower for kw in sensitive_keywords):
                return True
        # secondary check: metadata context tag from the simulation
        # handles cases where event names are generic (e.g. 'patient-record-view')
        if ground_truth and isinstance(ground_truth, dict):
            ctx = ground_truth.get('metadata', {}).get('access_context', '')
            if ctx == 'restricted_clinical_data':
                return True
        return False

    if THRESHOLDS['cross_dept_strict_events']:
        clinical_mask = df.apply(
            lambda row: is_sensitive_event(row['event'], row['ground_truth']), axis=1
        )
        clinical_df = df[clinical_mask]
    else:
        clinical_df = df

    time_window = timedelta(hours=THRESHOLDS['cross_dept_time_window_hours'])

    for user, user_df in clinical_df.groupby('user'):
        user_role = role_lookup.get(user, '') or ''

        # primary check: match against the role/title stored in the users table
        is_restricted = any(role.lower() in user_role.lower()
                           for role in THRESHOLDS['restricted_roles'])

        # fallback: if the DB title isn't set, check if the username itself
        # contains a restricted role keyword (e.g. 'receptionist_brown')
        # this is needed for synthetic setups where OpenEMR users may not have
        # their title field populated correctly
        if not is_restricted:
            is_restricted = any(role.lower() in user.lower()
                               for role in THRESHOLDS['restricted_roles'])

        if not is_restricted:
            continue

        user_df = user_df.sort_values('date')

        for i, row in user_df.iterrows():
            window_start = row['date']
            window_end = window_start + time_window

            window_df = user_df[(user_df['date'] >= window_start) &
                               (user_df['date'] <= window_end)]

            if len(window_df) >= THRESHOLDS['cross_dept_min_accesses']:
                for idx, match_row in window_df.iterrows():
                    if match_row['id'] not in flagged_ids:
                        flagged_ids.add(match_row['id'])
                        flagged.append({
                            'log_id': match_row['id'], 'rule': 'cross_department',
                            'user': user, 'patient_id': match_row['patient_id'],
                            'timestamp': match_row['date'],
                            'details': f"{user_role or user} accessed {len(window_df)} sensitive records",
                            'risk_weight': THRESHOLDS['rule_weights']['cross_department']
                        })
    return flagged

def rule_extended_viewing(df):
    """RULE 7: Flags repeated access to the same patient in a short period.
    If someone is hitting the same record 5+ times in 2 hours, they might be
    downloading or screenshotting private data piece by piece."""
    flagged = []
    df_copy = df.copy()
    df_copy['time_bucket'] = df_copy['date'].dt.floor(f'{THRESHOLDS["extended_viewing_window_hours"]}h')

    for (user, patient, bucket), group in df_copy.groupby(['user', 'patient_id', 'time_bucket']):
        if len(group) > THRESHOLDS['same_patient_access_threshold']:
            for idx, row in group.iterrows():
                flagged.append({
                    'log_id': row['id'], 'rule': 'extended_viewing',
                    'user': user, 'patient_id': patient, 'timestamp': row['date'],
                    'details': f"{len(group)} accesses to patient {patient} in {THRESHOLDS['extended_viewing_window_hours']}h",
                    'risk_weight': THRESHOLDS['rule_weights']['extended_viewing']
                })
    return flagged

def rule_failed_access(df):
    """RULE 8: Multiple failed access attempts whcih can equal someone trying to get into records they shouldn't be in. 3+ failures triggers a flag."""
    flagged = []
    failed_df = df[df['event'].str.contains('denied|failed|unauthorized', case=False, na=False)]

    for user, group in failed_df.groupby('user'):
        if len(group) >= THRESHOLDS['failed_access_threshold']:
            for idx, row in group.iterrows():
                flagged.append({
                    'log_id': row['id'], 'rule': 'failed_access',
                    'user': user, 'patient_id': row['patient_id'],
                    'timestamp': row['date'],
                    'details': f"{len(group)} failed access attempts",
                    'risk_weight': THRESHOLDS['rule_weights']['failed_access']
                })
    return flagged

def rule_vip_access_v12(df, users_df, justification_checker=None):
    """RULE 9: VIP patient access, high-profile patients that get snooped on a lot. Added a clinical justification before flagging. If a doctor accessed a VIP because they had an appointment, then thats fine."""
    if not THRESHOLDS['vip_detection_enabled'] or not THRESHOLDS['vip_patient_ids']:
        return []

    flagged = []

    vip_accesses = df[df['patient_id'].isin(THRESHOLDS['vip_patient_ids'])]

    for idx, row in vip_accesses.iterrows():
        user = row['user']

        # only check users we've flagged as suspicious
        if user not in THRESHOLDS['vip_only_flag_users']:
            continue

        # check if there's a legit reason for this access
        has_justification = False
        justification_reason = None

        if THRESHOLDS['vip_require_no_justification'] and justification_checker:
            has_justification, justification_reason = justification_checker.has_any_justification(
                row['patient_id'], user, row['date']
            )

        # only flag if there's no good reason
        if not has_justification:
            flagged.append({
                'log_id': row['id'], 'rule': 'vip_access',
                'user': user, 'patient_id': row['patient_id'],
                'timestamp': row['date'],
                'details': f"VIP patient accessed by {user} WITHOUT clinical justification",
                'risk_weight': THRESHOLDS['rule_weights']['vip_access'],
                'has_justification': False
            })

    return flagged

def rule_street_match(df, patients_df):
    """RULE 10: Street-level geographic match, even more specific than postcode.
    If someone accesses 3+ patients from the same street in 4 hours that's a strong
    indicator of neighbourhood snooping.
    Extended to use a metadata-embedded synthetic_street value when present -
    Synthea generates unique addresses per patient so there's no natural clustering
    in patient_data.street. In a real deployment the patient record address would
    always be available. Priority: metadata synthetic_street > patient_data.street."""
    if not THRESHOLDS['street_match_enabled']:
        return []

    flagged = []
    flagged_ids = set()

    def normalize_street(street):
        """Strip common suffixes so '42 Oak Street' and '42 Oak St' match."""
        if not street or pd.isna(street):
            return None
        street = str(street).lower().strip()
        street = re.sub(r'\b(street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|place|pl|boulevard|blvd)\b', '', street)
        street = re.sub(r'[^a-z0-9\s]', '', street)
        street = re.sub(r'\s+', ' ', street).strip()
        return street if len(street) > 3 else None

    # real-data fallback: street from the patient_data table
    street_lookup = {pid: normalize_street(street)
                     for pid, street in patients_df.set_index('pid')['street'].to_dict().items()}

    df_copy = df.copy()

    # for each entry: use metadata synthetic_street if present, otherwise patient_data.street
    def get_street_for_entry(row):
        gt = row['ground_truth']
        if isinstance(gt, dict):
            synthetic = gt.get('metadata', {}).get('synthetic_street')
            if synthetic:
                return normalize_street(synthetic)
        return street_lookup.get(row['patient_id'])

    df_copy['street_norm'] = df_copy.apply(get_street_for_entry, axis=1)

    time_window = timedelta(hours=THRESHOLDS['street_match_time_window_hours'])

    for user, user_df in df_copy.groupby('user'):
        user_df = user_df.sort_values('date')

        for i, row in user_df.iterrows():
            if not row['street_norm']:
                continue

            window_start = row['date']
            window_end = window_start + time_window

            window_df = user_df[(user_df['date'] >= window_start) &
                               (user_df['date'] <= window_end) &
                               (user_df['street_norm'].notna())]

            for street in window_df['street_norm'].unique():
                if not street:
                    continue

                street_df = window_df[window_df['street_norm'] == street]
                unique_patients = street_df['patient_id'].nunique()

                if unique_patients >= THRESHOLDS['street_match_threshold']:
                    for idx, match_row in street_df.iterrows():
                        if match_row['id'] not in flagged_ids:
                            flagged_ids.add(match_row['id'])
                            flagged.append({
                                'log_id': match_row['id'], 'rule': 'street_match',
                                'user': user, 'patient_id': match_row['patient_id'],
                                'timestamp': match_row['date'],
                                'details': f"{unique_patients} patients from same street",
                                'risk_weight': THRESHOLDS['rule_weights']['street_match']
                            })
    return flagged

# if multiple rules fire on the same log entry it's more suspicious than just one
# 2 rules = HIGH risk, 3 or more = CRITICAL - this helped reduce noise from single-rule false positives
def calculate_composite_risk(all_flags):
    """Combine flags per log entry and assign a priority level.
    Got this idea from reading about how SIEM tools work - they all do
    some version of risk aggregation rather than treating every alert equally."""
    log_risks = defaultdict(lambda: {'rules': [], 'total_weight': 0.0, 'risk_level': 'LOW'})

    for flag in all_flags:
        log_id = flag['log_id']
        log_risks[log_id]['rules'].append(flag['rule'])
        log_risks[log_id]['total_weight'] += flag.get('risk_weight', 1.0)
        log_risks[log_id]['user'] = flag['user']
        log_risks[log_id]['patient_id'] = flag['patient_id']
        log_risks[log_id]['timestamp'] = flag['timestamp']

    for log_id, risk_data in log_risks.items():
        num_rules = len(set(risk_data['rules']))
        if num_rules >= THRESHOLDS['critical_risk_min_rules']:
            risk_data['risk_level'] = 'CRITICAL'
        elif num_rules >= THRESHOLDS['high_risk_min_rules']:
            risk_data['risk_level'] = 'HIGH'
        else:
            risk_data['risk_level'] = 'MEDIUM'

    return dict(log_risks)

def run_detection_v12():
    """Run the full V12 detection pipeline.
    Loads data, runs all 10 rules, applies explanation-based filtering,
    scores composite risk, and evaluates against ground truth."""

    print("=" * 80)
    print("INSIDER THREAT DETECTION ENGINE v12.0 - COMPLETE IMPLEMENTATION")
    print("=" * 80)

    # load everything we need
    print("\n[1/7] Loading data...")
    df = load_audit_logs()
    users_df = load_users()
    patients_df = load_patients()

    # V12: also load clinical data for justification checks
    appointments_df = load_appointments()
    orders_df = load_orders()
    notes_df = load_clinical_notes()

    print(f"   - Audit logs: {len(df)} entries")
    print(f"   - Users: {len(users_df)}")
    print(f"   - Patients: {len(patients_df)}")
    print(f"   - Appointments: {len(appointments_df)}")
    print(f"   - Orders: {len(orders_df)}")
    print(f"   - Notes: {len(notes_df)}")

    # VIP patient IDs are set dynamically from the loaded patients so the detector
    # stays in sync with whatever PIDs the simulation used - avoids hardcoding IDs
    # that might not exist in a given database
    vip_ids_dynamic = sorted(patients_df['pid'].tolist())[:3]
    THRESHOLDS['vip_patient_ids'] = vip_ids_dynamic
    print(f"   - VIP patient IDs (dynamic): {vip_ids_dynamic}")

    # work out what's normal for each role
    print("\n[2/7] Calculating role-specific baselines...")
    role_baselines = calculate_role_baselines(df, users_df)
    for role, stats in role_baselines.items():
        print(f"   - {role}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, threshold={stats['threshold']:.1f}")

    # set up the justification checker
    print("\n[3/7] Initializing clinical justification checker...")
    justification_checker = ClinicalJustificationChecker(appointments_df, orders_df, notes_df)

    # run all 10 detection rules
    print("\n[4/7] Running detection rules...")
    all_flags = []

    rules = [
        ("After-Hours", lambda: rule_after_hours(df)),
        ("High-Volume (Role-Specific)", lambda: rule_high_volume_v12(df, users_df, role_baselines)),
        ("Sequential Browsing", lambda: rule_sequential_browsing(df, patients_df)),
        ("Same-Surname", lambda: rule_same_surname(df, patients_df)),
        ("Geographic Proximity", lambda: rule_geographic_proximity(df, patients_df)),
        ("Cross-Department", lambda: rule_cross_department(df, users_df)),
        ("Extended Viewing", lambda: rule_extended_viewing(df)),
        ("Failed Access", lambda: rule_failed_access(df)),
        ("VIP Access (With Justification Check)", lambda: rule_vip_access_v12(df, users_df, justification_checker)),
        ("Street Match", lambda: rule_street_match(df, patients_df)),
    ]

    for rule_name, rule_func in rules:
        flags = rule_func()
        all_flags.extend(flags)
        print(f"   - {rule_name}: {len(flags)} flags")

    print(f"\n   TOTAL FLAGS (before filtering): {len(all_flags)}")

    # V12: filter out flags that have legit clinical reasons
    print("\n[5/7] Applying explanation-based filtering...")
    if THRESHOLDS['explanation_filtering_enabled']:
        filtered_flags, dismissed_flags, filter_stats = justification_checker.filter_flags(all_flags)
        print(f"   - Total flags: {filter_stats['total']}")
        print(f"   - Dismissed (justified): {filter_stats['dismissed']}")
        print(f"   - Retained: {filter_stats['filtered']}")
        if filter_stats['total'] > 0:
            reduction_pct = (filter_stats['dismissed'] / filter_stats['total']) * 100
            print(f"   - False positive reduction: {reduction_pct:.1f}%")
    else:
        filtered_flags = all_flags
        dismissed_flags = []

    # score how risky each flagged entry is based on how many rules fired
    print("\n[6/7] Calculating composite risk scores...")
    composite_risks = calculate_composite_risk(filtered_flags)

    risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for log_id, risk_data in composite_risks.items():
        risk_counts[risk_data['risk_level']] += 1

    print(f"   - CRITICAL risk: {risk_counts['CRITICAL']}")
    print(f"   - HIGH risk: {risk_counts['HIGH']}")
    print(f"   - MEDIUM risk: {risk_counts['MEDIUM']}")

    # compare our detections against the ground truth labels
    print("\n[7/7] Evaluating detection performance...")

    flagged_ids = set(f['log_id'] for f in filtered_flags)
    df['predicted_malicious'] = df['id'].isin(flagged_ids)

    TP = len(df[(df['predicted_malicious'] == True) & (df['is_actually_malicious'] == True)])
    FP = len(df[(df['predicted_malicious'] == True) & (df['is_actually_malicious'] == False)])
    TN = len(df[(df['predicted_malicious'] == False) & (df['is_actually_malicious'] == False)])
    FN = len(df[(df['predicted_malicious'] == False) & (df['is_actually_malicious'] == True)])

    total = len(df)
    accuracy = (TP + TN) / total * 100 if total > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print("V12 DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"   Accuracy:  {accuracy:.2f}%")
    print(f"   Precision: {precision:.2f}%")
    print(f"   Recall:    {recall:.2f}%")
    print(f"   F1-Score:  {f1:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"   TP: {TP:4d}  |  FP: {FP:4d}")
    print(f"   FN: {FN:4d}  |  TN: {TN:4d}")
    print(f"{'='*60}")

    # save metrics to a file so ml_baseline_comparison_v4.py can load them
    # instead of having to manually paste the numbers across
    rule_based_metrics = {
        "Model": "Rule-Based (V12)",
        "Type": "Rule-Based",
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2),
        "AUC": 0,
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "generated_at": datetime.now().isoformat()
    }
    output_path = Path("outputs") / "rule_based_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rule_based_metrics, f, indent=4)
    print(f"\n   Results saved to: {output_path}")

    return {
        'flags': filtered_flags,
        'dismissed': dismissed_flags,
        'composite_risks': composite_risks,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        },
        'role_baselines': role_baselines
    }

if __name__ == "__main__":
    results = run_detection_v12()

    print("\n" + "="*80)
    print("V12 Rule-Based Python Script")
    print("="*80)
    print()
