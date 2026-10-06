import xml.etree.ElementTree as ET
import os
import glob
from sqlalchemy import create_engine, text
from datetime import datetime

# Database connection
DB_USER = "openemr"
DB_PASS = "openemrpass"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "openemr"

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# CCDA namespace
NS = {'hl7': 'urn:hl7-org:v3'}

def parse_ccda(filepath):
    """Parse one Synthea CCDA file and pull out the patient fields we need.
    Synthea sometimes appends numbers to names so there's a cleanup step for that."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Find patient element
        patient = root.find('.//hl7:patient', NS)
        if patient is None:
            return None

        # Extract name
        given = patient.find('.//hl7:given', NS)
        family = patient.find('.//hl7:family', NS)

        # Clean up names (remove numbers from Synthea names)
        fname = given.text.split()[0] if given is not None and given.text else "Unknown"
        # Remove trailing numbers from first name
        fname = ''.join([c for c in fname if not c.isdigit()])

        lname = family.text if family is not None and family.text else "Unknown"
        # Remove trailing numbers from last name
        lname = ''.join([c for c in lname if not c.isdigit()])

        # Extract gender
        gender_elem = patient.find('.//hl7:administrativeGenderCode', NS)
        gender = gender_elem.get('code') if gender_elem is not None else 'U'
        # Convert to OpenEMR format
        sex = 'Female' if gender == 'F' else 'Male' if gender == 'M' else 'Unassigned'

        # Extract DOB
        dob_elem = patient.find('.//hl7:birthTime', NS)
        dob_str = dob_elem.get('value') if dob_elem is not None else None
        if dob_str:
            # Parse YYYYMMDDHHMMSS format
            dob = datetime.strptime(dob_str[:8], '%Y%m%d').strftime('%Y-%m-%d')
        else:
            dob = '1990-01-01'

        # Extract address
        addr = root.find('.//hl7:patientRole/hl7:addr', NS)
        street = city = state = postal = ""
        if addr is not None:
            street_elem = addr.find('hl7:streetAddressLine', NS)
            city_elem = addr.find('hl7:city', NS)
            state_elem = addr.find('hl7:state', NS)
            postal_elem = addr.find('hl7:postalCode', NS)

            street = street_elem.text if street_elem is not None else ""
            city = city_elem.text if city_elem is not None else ""
            state = state_elem.text if state_elem is not None else ""
            postal = postal_elem.text if postal_elem is not None else ""

        return {
            'fname': fname,
            'lname': lname,
            'sex': sex,
            'DOB': dob,
            'street': street,
            'city': city,
            'state': state,
            'postal_code': postal
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def insert_patient(patient_data):
    """Insert a parsed patient record into OpenEMR's patient_data table."""
    query = text("""
        INSERT INTO patient_data
        (fname, lname, sex, DOB, street, city, state, postal_code, date)
        VALUES
        (:fname, :lname, :sex, :DOB, :street, :city, :state, :postal_code, NOW())
    """)

    with engine.connect() as conn:
        result = conn.execute(query, patient_data)
        conn.commit()
        return result.lastrowid

# Main import process
# update this to wherever you ran Synthea on your machine, the ccda folder
# will be inside synthea/output/ccda/ after you run it
ccda_folder = os.path.expanduser("~/fyp-ehr-detection/synthea/output/ccda/")
xml_files = glob.glob(os.path.join(ccda_folder, "*.xml"))

print(f"Found {len(xml_files)} CCDA files")
print("="*60)

imported = 0
failed = 0

# Import first 50 patients (you can change this number)
for filepath in xml_files[:50]:
    patient = parse_ccda(filepath)
    if patient:
        try:
            pid = insert_patient(patient)
            print(f"Imported: {patient['fname']} {patient['lname']} (PID: {pid})")
            imported += 1
        except Exception as e:
            print(f"Failed to insert {patient['fname']} {patient['lname']}: {e}")
            failed += 1
    else:
        failed += 1

print("="*60)
print(f"Import complete: {imported} patients imported, {failed} failed")
