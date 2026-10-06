from sqlalchemy import create_engine, text
import pandas as pd
import random
from datetime import datetime, timedelta
import json

# fixed seed so the simulation produces the same events every time it runs.
# useful for demos and marking - the exact entry count and timestamps won't
# vary between machines or runs. the main src/ version doesn't have this
# so each dissertation run was genuinely fresh data.
random.seed(42)

# fixed anchor date instead of datetime.now() - means the generated timestamps
# won't drift over time and the output stays reproducible regardless of when you run it
BASE_TIME = datetime(2026, 1, 1)

# thresholds that mirror the detection engine settings - kept in sync so the
# simulation generates patterns that actually trigger (or stay below) the right rules
CONFIG = {
    'normal_hours_start': 6,
    'normal_hours_end': 22,
    'after_hours_incidents_threshold': 3,
    'high_volume_patients_per_hour': 10,
    'max_normal_patients_per_session': 5,
    'sequential_access_window_minutes': 3,
    'same_surname_count_threshold': 2,
    'num_normal_events': 200,  # background traffic to give the rules proper context
    'num_days_to_simulate': 30,
    'clinical_roles': ['Physician', 'Nurse'],
    'non_clinical_roles': ['Receptionist', 'Administrator'],
}

# database connection config
DB_CONFIG = {
    'user': 'openemr',
    'password': 'openemrpass',
    'host': 'localhost',
    'port': '3306',
    'database': 'openemr'
}

engine = create_engine(
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

def load_patients():
    """Pull all patient records - need their names, postcodes and streets
    for generating the surname/geographic patterns."""
    query = """
        SELECT pid, fname, lname, DOB, sex, street, city, postal_code
        FROM patient_data
        WHERE pid > 0
    """
    return pd.read_sql(query, engine)

def load_users():
    """Load the test user accounts I set up in OpenEMR. Only pulls the ones
    I created for the simulation, not any real accounts."""
    query = """
        SELECT id, username, fname, lname, title
        FROM users
        WHERE username IN ('admin', 'dr_smith', 'nurse_jones', 'admin_lee',
                          'receptionist_brown', 'malicious_user')
        AND active = 1
    """
    return pd.read_sql(query, engine)

def insert_log_entry(conn, username, patient_id, event_type, access_time,
                     is_malicious, threat_pattern=None, metadata=None):
    """Inserts one access event into the audit log. Embeds the ground truth label
    as JSON in the comments field with a 'SIMULATED|' prefix so the detection
    engine and ML scripts can find and evaluate these entries later."""
    ground_truth = {
        'is_malicious': is_malicious,
        'label': 'MALICIOUS' if is_malicious else 'NORMAL',
        'threat_pattern': threat_pattern,
        'metadata': metadata or {}
    }

    comments = f"SIMULATED|{json.dumps(ground_truth)}"

    query = text("""
        INSERT INTO log (date, event, user, patient_id, success, comments, groupname)
        VALUES (:date, :event, :user, :patient_id, 1, :comments, 'Default')
    """)

    conn.execute(query, {
        'date': access_time.strftime('%Y-%m-%d %H:%M:%S'),
        'event': event_type,
        'user': username,
        'patient_id': patient_id,
        'comments': comments
    })

def simulate_normal_patterns(conn, users_df, patients_df):
    """Creates normal background access activity so the detection engine has
    legitimate traffic to work with - without this, everything would look suspicious.
    Skips most weekends and mixes up session times to make the pattern more realistic.
    malicious_user is excluded here since they only appear in the threat patterns."""
    print("\n" + "="*60)
    print("GENERATING NORMAL ACCESS PATTERNS")
    print("="*60)

    normal_users = users_df[users_df['username'] != 'malicious_user']
    patient_ids = patients_df['pid'].tolist()

    event_types = [
        'patient-record-view',
        'demographics-view',
        'encounter-view',
        'vitals-view',
        'medication-view',
        'allergy-view'
    ]

    events_created = 0

    for day in range(CONFIG['num_days_to_simulate']):
        current_date = BASE_TIME - timedelta(days=day)

        # skip most weekends but not all - some staff do work weekends
        if current_date.weekday() >= 5 and random.random() < 0.6:
            continue

        for _, user in normal_users.iterrows():
            num_sessions = random.randint(2, 4)

            for session in range(num_sessions):
                # 70% chance of core hours, 30% chance of early/late to mix it up
                if random.random() < 0.7:
                    hour = random.choice([9, 10, 11, 14, 15, 16])
                else:
                    hour = random.choice([7, 8, 12, 13, 17, 18, 19])

                session_start = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )

                num_patients = random.randint(1, CONFIG['max_normal_patients_per_session'])
                session_patients = random.sample(patient_ids, min(num_patients, len(patient_ids)))

                for i, pid in enumerate(session_patients):
                    access_time = session_start + timedelta(minutes=i * random.randint(2, 8))
                    event = random.choice(event_types)

                    insert_log_entry(
                        conn=conn,
                        username=user['username'],
                        patient_id=pid,
                        event_type=event,
                        access_time=access_time,
                        is_malicious=False,
                        threat_pattern=None,
                        metadata={
                            'user_role': user['title'],
                            'session_id': f"{user['username']}_{day}_{session}",
                            'access_context': 'clinical_workflow'
                        }
                    )
                    events_created += 1

    print(f"Created {events_created} normal access events")
    print(f"  Users: {', '.join(normal_users['username'].tolist())}")
    print(f"  Time range: Past {CONFIG['num_days_to_simulate']} days")
    print(f"  Hours: {CONFIG['normal_hours_start']}:00 - {CONFIG['normal_hours_end']}:00")

    return events_created

def pattern_after_hours(conn, user, patient_ids):
    """Generates after-hours access events, spread across random nights and early
    mornings over the past two weeks. Hours are picked from 10pm to 5am."""
    print("\n  [Pattern 1] After-Hours Access")
    events = 0

    for _ in range(20):
        pid = random.choice(patient_ids)
        days_ago = random.randint(1, 14)

        hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
        access_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
        )

        insert_log_entry(
            conn=conn,
            username=user['username'],
            patient_id=pid,
            event_type='patient-record-view',
            access_time=access_time,
            is_malicious=True,
            threat_pattern='after_hours',
            metadata={
                'access_hour': hour,
                'rule': f'outside_{CONFIG["normal_hours_start"]}AM_{CONFIG["normal_hours_end"]}PM'
            }
        )
        events += 1

    print(f"    Created {events} after-hours access events")
    return events

def pattern_high_volume(conn, user, patient_ids):
    """Generates burst access sessions where the user hits 20-30 patients
    in under an hour, well over the 10/hr threshold. Creates 3 separate sessions."""
    print("\n  [Pattern 2] High-Volume Access")
    events = 0

    for session in range(3):
        days_ago = random.randint(2, 10)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(10, 15), minute=0, second=0
        )

        num_patients = random.randint(20, 30)
        session_patients = random.sample(patient_ids, min(num_patients, len(patient_ids)))

        for i, pid in enumerate(session_patients):
            access_time = base_time + timedelta(minutes=i * 2)

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=pid,
                event_type='patient-record-view',
                access_time=access_time,
                is_malicious=True,
                threat_pattern='high_volume',
                metadata={
                    'session_id': session,
                    'patients_accessed': num_patients,
                    'rule': f'>{CONFIG["high_volume_patients_per_hour"]}_patients_per_hour'
                }
            )
            events += 1

    print(f"    Created {events} high-volume access events (3 sessions)")
    return events

def pattern_sequential_browsing(conn, user, patients_df):
    """Simulates someone browsing through the patient list alphabetically -
    takes a slice of patients sorted by surname and accesses them in order
    a few minutes apart. Does this twice to create 2 distinct sequences."""
    print("\n  [Pattern 3] Sequential Alphabetical Browsing")
    events = 0

    sorted_patients = patients_df.sort_values('lname')

    for session in range(2):
        start_idx = random.randint(0, max(0, len(sorted_patients) - 25))
        sequence = sorted_patients.iloc[start_idx:start_idx + 20]

        days_ago = random.randint(3, 12)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(9, 14), minute=0, second=0
        )

        for i, (_, patient) in enumerate(sequence.iterrows()):
            access_time = base_time + timedelta(minutes=i * random.randint(2, 3))

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=patient['pid'],
                event_type='patient-record-view',
                access_time=access_time,
                is_malicious=True,
                threat_pattern='sequential_browse',
                metadata={
                    'sequence_position': i,
                    'patient_surname': patient['lname'],
                    'rule': 'alphabetical_access_pattern'
                }
            )
            events += 1

    print(f"    Created {events} sequential browsing events (2 sequences)")
    return events

def pattern_same_surname(conn, user, patients_df):
    """Accesses multiple patients who share the same surname, one after another.
    Uses up to 5 different surnames that have 2+ patients in the dataset.
    This is the 'snooping on a family' pattern from the Hedda et al. paper."""
    print("\n  [Pattern 4] Same-Surname Snooping")
    events = 0

    surname_counts = patients_df['lname'].value_counts()
    duplicate_surnames = surname_counts[surname_counts >= 2].index.tolist()

    if not duplicate_surnames:
        print("    No duplicate surnames found - skipping")
        return 0

    for surname in duplicate_surnames[:5]:
        same_name_patients = patients_df[patients_df['lname'] == surname]

        days_ago = random.randint(5, 15)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(10, 14), minute=random.randint(0, 30)
        )

        for i, (_, patient) in enumerate(same_name_patients.iterrows()):
            access_time = base_time + timedelta(minutes=i * random.randint(3, 7))

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=patient['pid'],
                event_type='patient-record-view',
                access_time=access_time,
                is_malicious=True,
                threat_pattern='same_surname',
                metadata={
                    'surname': surname,
                    'same_surname_count': len(same_name_patients),
                    'rule': f'>={CONFIG["same_surname_count_threshold"]}_same_surname'
                }
            )
            events += 1

    print(f"    Created {events} same-surname snooping events")
    return events

def pattern_geographic_proximity(conn, user, patients_df):
    """Accesses multiple patients from the same postcode area - the
    'neighbour snooping' pattern. Finds postcodes that have 3+ patients
    and generates clustered accesses for up to 3 of them."""
    print("\n  [Pattern 5] Geographic Proximity Access")
    events = 0

    postal_counts = patients_df['postal_code'].value_counts()
    common_postcodes = postal_counts[postal_counts >= 3].index.tolist()

    if not common_postcodes:
        print("    No common postal codes found - skipping")
        return 0

    for postcode in common_postcodes[:3]:
        nearby_patients = patients_df[patients_df['postal_code'] == postcode]

        days_ago = random.randint(4, 12)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(11, 16), minute=random.randint(0, 30)
        )

        for i, (_, patient) in enumerate(nearby_patients.head(5).iterrows()):
            access_time = base_time + timedelta(minutes=i * random.randint(4, 8))

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=patient['pid'],
                event_type='patient-record-view',
                access_time=access_time,
                is_malicious=True,
                threat_pattern='geographic_proximity',
                metadata={
                    'postal_code': postcode,
                    'patients_in_area': len(nearby_patients),
                    'rule': 'same_postal_code_cluster'
                }
            )
            events += 1

    print(f"    Created {events} geographic proximity events")
    return events

def pattern_cross_department(conn, users_df, patient_ids):
    """Receptionist accessing clinical records they have no business reason to view.
    Events are generated in clustered windows (8 per window) so the detection engine's
    threshold of 6 accesses in 3 hours is reliably met. Previously the events were
    randomly spread across 20 days which made it unlikely for 6 to land in the same
    3-hour window, so the rule never fired consistently.
    Also adds access_context='restricted_clinical_data' to metadata so the detector
    can identify sensitive access even if the event name doesn't contain a keyword."""
    print("\n  [Pattern 6] Cross-Department Access")
    events = 0

    receptionist = users_df[users_df['username'] == 'receptionist_brown']
    if receptionist.empty:
        print("    Receptionist user not found - skipping")
        return 0
    receptionist = receptionist.iloc[0]

    clinical_events = ['medication-view', 'diagnosis-view', 'lab-results-view', 'clinical-notes-view']

    # generate 2 clustered windows of 8 events each - ensures 6+ in 3 hours threshold is met
    for window in range(2):
        days_ago = random.randint(1, 20)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(9, 13), minute=0, second=0
        )

        window_patients = random.sample(patient_ids, min(8, len(patient_ids)))

        for i, pid in enumerate(window_patients):
            # space events ~12 mins apart, keeps all 8 within 2 hours
            access_time = base_time + timedelta(minutes=i * random.randint(10, 15))

            insert_log_entry(
                conn=conn,
                username=receptionist['username'],
                patient_id=pid,
                event_type=random.choice(clinical_events),
                access_time=access_time,
                is_malicious=True,
                threat_pattern='cross_department',
                metadata={
                    'user_role': 'Receptionist',
                    'access_context': 'restricted_clinical_data',
                    'expected_access': 'demographics_only',
                    'rule': 'role_inappropriate_access'
                }
            )
            events += 1

    print(f"    Created {events} cross-department access events (2 clustered windows of 8)")
    return events

def pattern_extended_viewing(conn, user, patient_ids):
    """Repeatedly accesses the same patient record many times in a short period -
    10 to 15 hits on 5 different patients. Could indicate someone is copying
    data piece by piece or screenshotting sections of a record."""
    print("\n  [Pattern 7] Extended Viewing Duration")
    events = 0

    target_patients = random.sample(patient_ids, min(5, len(patient_ids)))

    for pid in target_patients:
        days_ago = random.randint(2, 15)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(10, 15), minute=random.randint(0, 30)
        )

        num_views = random.randint(10, 15)
        view_types = ['patient-record-view', 'demographics-view', 'encounter-view',
                     'medication-view', 'allergy-view', 'lab-results-view']

        for i in range(num_views):
            access_time = base_time + timedelta(minutes=i * random.randint(1, 3))

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=pid,
                event_type=random.choice(view_types),
                access_time=access_time,
                is_malicious=True,
                threat_pattern='extended_viewing',
                metadata={
                    'view_count': num_views,
                    'total_duration_mins': num_views * 2,
                    'rule': 'repeated_same_patient_access'
                }
            )
            events += 1

    print(f"    Created {events} extended viewing events")
    return events

def pattern_failed_access(conn, user, patient_ids):
    """Generates repeated denied/failed access events - someone trying to get into
    records they don't have permission for. 3 sessions of 4-6 attempts each."""
    print("\n  [Pattern 8] Failed Access Attempts")
    events = 0

    for session in range(3):
        days_ago = random.randint(1, 10)
        base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(8, 20), minute=random.randint(0, 30)
        )

        num_attempts = random.randint(4, 6)

        for i in range(num_attempts):
            pid = random.choice(patient_ids)
            access_time = base_time + timedelta(minutes=i * random.randint(1, 3))

            insert_log_entry(
                conn=conn,
                username=user['username'],
                patient_id=pid,
                event_type='patient-access-denied',
                access_time=access_time,
                is_malicious=True,
                threat_pattern='failed_access',
                metadata={
                    'attempt_number': i + 1,
                    'session_attempts': num_attempts,
                    'rule': 'repeated_failed_attempts'
                }
            )
            events += 1

    print(f"    Created {events} failed access attempt events")
    return events

def pattern_street_match(conn, user, patient_ids):
    """Accesses multiple patients who share the same street address - the most
    specific geographic clustering pattern. Since Synthea assigns a unique address
    to every synthetic patient, there's no natural street clustering in patient_data.
    Instead we embed a synthetic common street value in the metadata so the
    detection engine can identify the cluster. The detector is updated to use
    this field when present, falling back to patient_data.street on real data."""
    print("\n  [Pattern 9] Street-Level Geographic Match")
    events = 0

    synthetic_street = '42 Maple Street'
    # pick 4 patients to cluster on this street - 3+ triggers the rule
    street_patients = random.sample(patient_ids, min(4, len(patient_ids)))

    days_ago = random.randint(3, 12)
    base_time = (BASE_TIME - timedelta(days=days_ago)).replace(
        hour=random.randint(10, 15), minute=0, second=0
    )

    for i, pid in enumerate(street_patients):
        access_time = base_time + timedelta(minutes=i * random.randint(25, 45))

        insert_log_entry(
            conn=conn,
            username=user['username'],
            patient_id=pid,
            event_type='patient-record-view',
            access_time=access_time,
            is_malicious=True,
            threat_pattern='street_match',
            metadata={
                'synthetic_street': synthetic_street,
                'rule': 'same_street_cluster'
            }
        )
        events += 1

    print(f"    Created {events} street-match events (synthetic street: '{synthetic_street}')")
    return events


def pattern_vip_access(conn, user, patients_df):
    """malicious_user accessing high-profile patients without any clinical justification.
    VIP patient IDs are taken as the 3 lowest PIDs in the dataset - this mirrors what
    detection_engine_v12.py does at runtime (it dynamically sets vip_patient_ids to the
    same 3 lowest PIDs), keeping simulation and detector in sync without hardcoding
    specific IDs that might not exist in a given database."""
    print("\n  [Pattern 10] VIP Patient Access")
    events = 0

    # sort ascending and take first 3 - matches the dynamic VIP ID logic in the engine
    vip_pids = sorted(patients_df['pid'].tolist())[:3]

    for pid in vip_pids:
        days_ago = random.randint(1, 14)
        access_time = (BASE_TIME - timedelta(days=days_ago)).replace(
            hour=random.randint(9, 17), minute=random.randint(0, 59), second=0
        )

        insert_log_entry(
            conn=conn,
            username=user['username'],
            patient_id=pid,
            event_type='patient-record-view',
            access_time=access_time,
            is_malicious=True,
            threat_pattern='vip_access',
            metadata={
                'vip_patient_id': int(pid),
                'rule': 'high_profile_patient_access'
            }
        )
        events += 1

    print(f"    Created {events} VIP access events (PIDs: {vip_pids})")
    return events


def main():
    """Runs the full simulation. Clears any previous simulation data first so
    you always start clean, then generates normal background traffic followed
    by all 10 malicious threat patterns. Everything is committed together at the end."""

    print("="*70)
    print("OPENEMR INSIDER THREAT DETECTION - ACCESS PATTERN SIMULATION")
    print("="*70)

    patients_df = load_patients()
    users_df = load_users()
    patient_ids = patients_df['pid'].tolist()

    print(f"\nData loaded:")
    print(f"  Patients: {len(patients_df)}")
    print(f"  Users: {len(users_df)}")
    print(f"  User list: {', '.join(users_df['username'].tolist())}")

    malicious_user = users_df[users_df['username'] == 'malicious_user']
    if malicious_user.empty:
        print("ERROR: malicious_user not found!")
        return
    malicious_user = malicious_user.iloc[0]

    with engine.connect() as conn:
        # wipe previous simulation data so we start fresh each run
        conn.execute(text("DELETE FROM log WHERE comments LIKE 'SIMULATED|%'"))
        conn.commit()
        print("\nCleared previous simulation data")

        normal_count = simulate_normal_patterns(conn, users_df, patients_df)

        print("\n" + "="*60)
        print("GENERATING MALICIOUS ACCESS PATTERNS")
        print("="*60)

        malicious_count = 0
        malicious_count += pattern_after_hours(conn, malicious_user, patient_ids)
        malicious_count += pattern_high_volume(conn, malicious_user, patient_ids)
        malicious_count += pattern_sequential_browsing(conn, malicious_user, patients_df)
        malicious_count += pattern_same_surname(conn, malicious_user, patients_df)
        malicious_count += pattern_geographic_proximity(conn, malicious_user, patients_df)
        malicious_count += pattern_cross_department(conn, users_df, patient_ids)
        malicious_count += pattern_extended_viewing(conn, malicious_user, patient_ids)
        malicious_count += pattern_failed_access(conn, malicious_user, patient_ids)
        malicious_count += pattern_street_match(conn, malicious_user, patient_ids)
        malicious_count += pattern_vip_access(conn, malicious_user, patients_df)

        conn.commit()

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Normal events:     {normal_count:>6}")
    print(f"Malicious events:  {malicious_count:>6}")
    print(f"Total events:      {normal_count + malicious_count:>6}")
    print("-"*70)
    print("Ground truth labels stored in 'comments' field as JSON")
    print("\nThreat patterns simulated (all 10 rules covered):")
    print("  1. After-hours access")
    print("  2. High-volume access")
    print("  3. Sequential alphabetical browsing")
    print("  4. Same-surname snooping")
    print("  5. Geographic proximity access")
    print("  6. Cross-department access")
    print("  7. Extended viewing duration")
    print("  8. Failed access attempts")
    print("  9. Street-level geographic match")
    print(" 10. VIP patient access")
    print("="*70)

    return normal_count, malicious_count

if __name__ == "__main__":
    main()
