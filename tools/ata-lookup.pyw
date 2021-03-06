#!/usr/bin/env python3

# Small standalone program for looking up ATA codes using a Tkinter GUI.
# Last updated: 7 Mar 2020 by Eric J. Whitney

import tkinter as tk

ATA_CODES_ = {
    '05': {'': 'TIME LIMITS / MAINTENANCE CHECKS',
           '00': 'General',
           '10': 'Time Limits',
           '20': 'Scheduled Maintenance Checks',
           '30': 'Reserved', '40': 'Reserved',
           '50': 'Unscheduled Maintenance Checks'},
    '06': {'': 'DIMENSIONS AND AREAS'},
    '07': {'': 'LIFTING & SHORING',
           '00': 'General',
           '10': 'Jacking',
           '20': 'Shoring'},
    '08': {'': 'LEVELING & WEIGHING',
           '00': 'General',
           '10': 'Weighing and Balancing',
           '20': 'Leveling'},
    '09': {'': 'TOWING & TAXIING',
           '00': 'General',
           '10': 'Towing',
           '20': 'Taxiing'},
    '10': {'': 'PARKING, MOORING, STORAGE & RETURN TO SERVICE',
           '00': 'General',
           '10': 'Parking/Storage',
           '20': 'Mooring',
           '30': 'Return to Service'},
    '12': {'': 'SERVICING',
           '00': 'General',
           '10': 'Replenishing',
           '20': 'Scheduled Servicing',
           '30': 'Unscheduled Servicing'},
    '18': {'': 'VIBRATION AND NOISE ANALYSIS (HELICOPTER ONLY)',
           '00': 'General',
           '10': 'Vibration Analysis',
           '20': 'Noise Analysis'},
    '20': {'': 'STANDARD PRACTICES-AIRFRAME'},
    '21': {'': 'AIR CONDITIONING',
           '00': 'General',
           '10': 'Compression',
           '20': 'Distribution',
           '30': 'Pressurization Control',
           '40': 'Heating',
           '50': 'Cooling',
           '60': 'Temperature Control',
           '70': 'Moisture/Air Contaminant Control'},
    '22': {'': 'AUTO FLIGHT',
           '00': 'General',
           '10': 'Autopilot',
           '20': 'Speed-Attitude Correction',
           '30': 'Auto Throttle',
           '40': 'System Monitor',
           '50': 'Aerodynamic Load Alleviating'},
    '23': {'': 'COMMUNICATIONS',
           '00': 'General',
           '10': 'Speech',
           '15': 'SATCOM',
           '20': 'Data Transmission and Automatic Calling',
           '30': 'Passenger Address, Entertainment and Comfort',
           '40': 'Interphone',
           '50': 'Audio Integrating',
           '60': 'Static Discharging',
           '70': 'Audio & Video Monitoring',
           '80': 'Integrated Automatic Tuning'},
    '24': {'': 'ELECTRICAL POWER',
           '00': 'General',
           '10': 'Generator Drive',
           '20': 'AC Generation',
           '30': 'DC Generation',
           '40': 'External Power',
           '50': 'AC Electrical Load',
           '60': 'DC Electrical Load Distribution'},
    '25': {'': 'EQUIPMENT/FURNISHINGS',
           '00': 'General',
           '10': 'Flight Compartment',
           '20': 'Passenger Compartment',
           '30': 'Galley',
           '40': 'Lavatories',
           '50': 'Additional Compartments',
           '60': 'Emergency',
           '70': 'Available',
           '80': 'Insulation'},
    '26': {'': 'FIRE PROTECTION',
           '00': 'General',
           '10': 'Detection',
           '20': 'Extinguishing',
           '30': 'Explosion Suppression'},
    '27': {'': 'FLIGHT CONTROLS',
           '00': 'General',
           '10': 'Aileron & Tab',
           '20': 'Rudder & Tab',
           '30': 'Elevator & Tab',
           '40': 'Horizontal Stabilizer',
           '50': 'Flaps',
           '60': 'Spoiler, Drag Devices and Variable Aerodynamic Fairings',
           '70': 'Gust Lock & Dampener',
           '80': 'Lift Augmenting'},
    '28': {'': 'FUEL',
           '00': 'General',
           '10': 'Storage',
           '20': 'Distribution',
           '30': 'Dump',
           '40': 'Indicating'},
    '29': {'': 'HYDRAULIC POWER',
           '00': 'General',
           '10': 'Main',
           '20': 'Auxiliary',
           '30': 'Indicating'},
    '30': {'': 'ICE AND RAIN PROTECTION',
           '00': 'General',
           '10': 'Airfoil',
           '20': 'Air Intakes',
           '30': 'Pitot and Static',
           '40': 'Windows, Windshields and Doors',
           '50': 'Antennas and Radomes',
           '60': 'Propellers/Rotors',
           '70': 'Water Lines',
           '80': 'Detection'},
    '31': {'': 'INDICATING/RECORDING SYSTEMS',
           '00': 'General',
           '10': 'Instrument & Control Panels',
           '20': 'Independent Instruments',
           '30': 'Recorders',
           '40': 'Central Computers',
           '50': 'Central Warning Systems',
           '60': 'Central Display Systems',
           '70': 'Automatic Data Reporting Systems'},
    '32': {'': 'LANDING GEAR',
           '00': 'General',
           '10': 'Main Gear and Doors',
           '20': 'Nose Gear and Doors',
           '30': 'Extension and Retraction',
           '40': 'Wheels and Brakes',
           '50': 'Steering',
           '60': 'Position and Warning',
           '70': 'Supplementary Gear Devices'},
    '33': {'': 'LIGHTS',
           '00': 'General',
           '10': 'Flight Compartment',
           '20': 'Passenger Compartment',
           '30': 'Cargo and Service Compartments',
           '40': 'Exterior',
           '50': 'Emergency Lighting'},
    '34': {'': 'NAVIGATION',
           '00': 'General',
           '10': 'Flight Environment Data',
           '20': 'Attitude & Direction',
           '30': 'Landing and Taxiing Aids T',
           '40': 'Independent Position Determing',
           '50': 'Dependent Position Determining',
           '60': 'Flight Management Computing'},
    '35': {'': 'OXYGEN',
           '00': 'General',
           '10': 'Crew',
           '20': 'Passenger',
           '30': 'Portable'},
    '36': {'': 'PNEUMATIC',
           '00': 'General',
           '10': 'Distribution',
           '20': 'Indicating'},
    '37': {'': 'VACUUM',
           '00': 'General',
           '10': 'Distribution',
           '20': 'Indicating'},
    '38': {'': 'WATER/WASTE',
           '00': 'General',
           '10': 'Potable',
           '20': 'Wash',
           '30': 'Waste Disposal',
           '40': 'Air Supply'},
    '41': {'': 'WATER BALLAST',
           '00': 'General',
           '10': 'Storage',
           '20': 'Dump',
           '30': 'Indication'},
    '44': {'': 'CABIN SYSTEMS',
           '00': 'General',
           '10': 'Cabin Core System',
           '20': 'Inflight Entertainment System',
           '30': 'External Communication System',
           '40': 'Cabin Mass Memory System',
           '50': 'Cabin Monitoring System',
           '60': 'Miscellaneous Cabin System'},
    '45': {'': 'CENTRAL MAINTENANCE SYSTEM (CMS)',
           '00': 'General',
           '05': 'CMS/Aircraft General',
           '20': 'CMS/Airframe Systems',
           '45': 'Central Maintenance System',
           '50': 'CMS/Structures',
           '60': 'CMS/Propellers',
           '70': 'CMS/Power Plant'},
    '46': {'': 'INFORMATION SYSTEMS',
           '00': 'General',
           '10': 'Airplane General Information Systems',
           '20': 'Flight Deck Information Systems',
           '30': 'Maintenance Information Systems',
           '40': 'Passenger Cabin Information Systems',
           '50': 'Miscellaneous Information Systems'},
    '49': {'': 'AIRBORNE AUXILIARY POWER',
           '00': 'General',
           '10': 'Power Plant',
           '20': 'Engine',
           '30': 'Engine Fuel and Control',
           '40': 'Ignition/Starting',
           '50': 'Air',
           '60': 'Engine Controls',
           '70': 'Indicating',
           '80': 'Exhaust',
           '90': 'Oil'},
    '50': {'': 'CARGO AND ACCESSORY COMPARTMENTS',
           '00': 'General',
           '10': 'Cargo Compartments',
           '20': 'Cargo Loading Systems',
           '30': 'Cargo Related Systems',
           '40': 'Available',
           '50': 'Accessory Compartments',
           '60': 'Insulation'},
    '51': {'': 'STANDARD PRACTICES AND STRUCTURES - GENERAL',
           '00': 'General',
           '10': 'Investigation, Cleanup and Aerodynamic Smoothness',
           '20': 'Processes',
           '30': 'Materials',
           '40': 'Fasteners',
           '50': 'Support of Airplane for Repair and Alignment Check '
                 'Procedures',
           '60': 'Control-Surface Balancing',
           '70': 'Repairs',
           '80': 'Electrical Bonding'},
    '52': {'': 'DOORS',
           '00': 'General',
           '10': 'Passenger/Crew Doors',
           '20': 'Emergency Exit',
           '30': 'Cargo',
           '40': 'Service and Miscellaneous',
           '50': 'Fixed Interior',
           '60': 'Entrance Stairs',
           '70': 'Monitoring and Operation',
           '80': 'Landing Gear'},
    '53': {'': 'FUSELAGE',
           '00': 'General',
           '10': 'Fuselage Section (As Required)',
           '20': 'Fuselage Section (As Required)',
           '30': 'Fuselage Section (As Required)'},
    '54': {'': 'NACELLES/PYLONS',
           '00': 'General',
           '10': 'Nacelle Section (As Required)',
           '20': 'Nacelle Section (As Required)',
           '30': 'Nacelle Section (As Required)',
           '50': 'Pylon (As Required)',
           '60': 'Pylon (As Required)',
           '70': 'Pylon (As Required)'},
    '55': {'': 'STABILIZERS',
           '00': 'General',
           '10': 'Horizontal Stabilizer or Canard',
           '20': 'Elevator',
           '30': 'Vertical Stabilizer',
           '40': 'Rudder'},
    '56': {'': 'WINDOWS',
           '00': 'General',
           '10': 'Flight Compartment',
           '20': 'Passenger Compartment',
           '30': 'Door',
           '40': 'Inspection and Observation'},
    '57': {'': 'WINGS',
           '00': 'General',
           '10': 'Center Wing',
           '20': 'Outer Wing',
           '30': 'Wing Tip',
           '40': 'Leading Edge and Leading Edge Devices',
           '50': 'Trailing Edge Trailing Edge Devices',
           '60': 'Ailerons and Elevons',
           '70': 'Spoilers',
           '80': '(as required)',
           '90': 'Wing Folding System'},
    '60': {'': 'STANDARD PRACTICES - PROPELLER/ROTOR'},
    '61': {'': 'PROPELLERS/PROPULSORS',
           '00': 'General',
           '10': 'Propeller Assembly',
           '20': 'Controlling',
           '30': 'Braking',
           '40': 'Indicating',
           '50': 'Propulsor Duct'},
    '62': {'': 'ROTOR',
           '00': 'General',
           '10': 'Rotor Blades',
           '20': 'Rotor Head',
           '30': 'Rotor Shaft',
           '40': 'Indicating'},
    '63': {'': 'ROTOR DRIVES',
           '00': 'General',
           '10': 'Engine/Gearbox Couplings',
           '20': 'Gearboxes',
           '30': 'Mounts, Attachments',
           '40': 'Indicating'},
    '64': {'': 'TAIL ROTOR',
           '00': 'General',
           '10': 'Rotor Blades',
           '20': 'Rotor Head',
           '30': 'Available',
           '40': 'Indicating'},
    '65': {'': 'TAIL ROTOR DRIVE',
           '00': 'General',
           '10': 'Shafts',
           '20': 'Gearboxes',
           '30': 'Available',
           '40': 'Indicating'},
    '66': {'': 'FOLDING BLADES/PYLON',
           '00': 'General',
           '10': 'Rotor Blades',
           '20': 'Tail Pylon',
           '30': 'Controls and Indicating'},
    '67': {'': 'ROTORS FLIGHT CONTROL',
           '00': 'General',
           '10': 'Rotor Control',
           '20': 'Anti-Torque Rotor Control (Yaw Control)',
           '30': 'Servo-Control System'},
    '70': {'': 'STANDARD PRACTICES - ENGINES'},
    '71': {'': 'POWER PLANT',
           '00': 'General',
           '10': 'Cowling',
           '20': 'Mounts',
           '30': 'Fireseals',
           '40': 'Attach Fittings',
           '50': 'Electrical Harness',
           '60': 'Air Intakes',
           '70': 'Engine Drains'},
    '72T': {'': 'ENGINE - TURBINE/TURBO PROP - DUCTED FAN/UNDUCTED FAN',
            '00': 'General',
            '10': 'Reduction Gear, Shaft Section',
            '20': 'Air Inlet Section',
            '30': 'Compressor Section',
            '40': 'Combustion Section',
            '50': 'Turbine Section',
            '60': 'Accessory Drives',
            '70': 'Bypass Section',
            '80': 'Propulsor Section'},
    '72R': {'': 'ENGINE - RECIPROCATING',
            '00': 'General',
            '10': 'Front Section',
            '20': 'Power Section',
            '30': 'Cylinder Section',
            '40': 'Supercharger Section',
            '50': 'Lubrication'},
    '73': {'': 'ENGINE FUEL AND CONTROL',
           '00': 'General',
           '10': 'Distribution',
           '20': 'Controlling',
           '30': 'Indicating'},
    '74': {'': 'IGNITION',
           '00': 'General',
           '10': 'Electrical Power Supply',
           '20': 'Distribution',
           '30': 'Switching'},
    '75': {'': 'AIR',
           '00': 'General',
           '10': 'Engine Anti-Icing',
           '20': 'Cooling',
           '30': 'Compressor Control',
           '40': 'Indicating'},
    '76': {'': 'ENGINE CONTROLS',
           '00': 'General',
           '10': 'Power Control',
           '20': 'Emergency Shutdown'},
    '77': {'': 'ENGINE INDICATING',
           '00': 'General',
           '10': 'Power',
           '20': 'Temperature',
           '30': 'Analyzers',
           '40': 'Integrated Engine Instrument Systems'},
    '78': {'': 'EXHAUST',
           '00': 'General',
           '10': 'Collector/Nozzle',
           '20': 'Noise Suppressor',
           '30': 'Thrust Reverser',
           '40': 'Supplementary Air'},
    '79': {'': 'OIL',
           '00': 'General',
           '10': 'Storage',
           '20': 'Distribution',
           '30': 'Indicating'},
    '80': {'': 'STARTING',
           '00': 'General',
           '10': 'Cranking'},
    '81': {'': 'TURBINES',
           '00': 'General',
           '10': 'Power Recovery',
           '20': 'Turbo-Supercharger'},
    '82': {'': 'WATER INJECTION',
           '00': 'General',
           '10': 'Storage',
           '20': 'Distribution',
           '30': 'Dumping and Purging',
           '40': 'Indicating'},
    '83': {'': 'ACCESSORY GEAR-BOXES',
           '00': 'General',
           '10': 'Drive Shaft Section',
           '20': 'Gearbox Section'},
    '84': {'': 'PROPULSION AUGMENTATION',
           '00': 'General',
           '10': 'Jet Assist Takeoff'},
    '91': {'': 'CHARTS'},
    '97': {'': 'WIRING REPORTING'},
    '115': {'': 'FLIGHT SIMULATOR SYSTEMS'},
    '116': {'': 'FLIGHT SIMULATOR CUING SYSTEMS'}
}

# Dict rebuilt as list at startup.
ATA_CODES = []
for chap, block in ATA_CODES_.items():
    for sec, title in block.items():
        ATA_CODES.append((chap, sec, title))


# ----------------------------------------------------------------------------

class ATACodeGUI(tk.Frame):
    def __init__(self):
        super().__init__()

        self.master.title('ATA LOOKUP')
        self.master.resizable(height=False, width=False)
        self.pack(fill='both', expand=True)

        tk.Label(self, text='CHAPTER').grid(row=0, column=0)
        self.chap_entry = tk.Entry(self, justify='center', width=4)
        self.chap_entry.grid(row=1, column=0)
        self.chap_entry.bind('<KeyRelease>', lambda e: self.keypress())

        tk.Label(self, text='-').grid(row=0, column=1)
        tk.Label(self, text='-').grid(row=1, column=1)

        tk.Label(self, text='SECTION').grid(row=0, column=2)
        self.sec_entry = tk.Entry(self, justify='center', width=4)
        self.sec_entry.grid(row=1, column=2)
        self.sec_entry.bind('<KeyRelease>', lambda e: self.keypress())

        tk.Label(self, text='TITLE').grid(row=0, column=3)
        self.title_entry = tk.Entry(self, justify='center', width=40)
        self.title_entry.grid(row=1, column=3)
        self.title_entry.bind('<KeyRelease>', lambda e: self.keypress())

        tk.Label(self, text="▼ FOUND ▼").grid(row=2, columnspan=4)

        self.result_text = tk.Text(self, height=30, width=60)
        self.result_text.grid(row=3, columnspan=4)
        self.result_text.config(wrap=tk.WORD, state=tk.DISABLED)

        self.keypress()  # Trigger first result display.

    # ------------------------------------------------------------------------

    def keypress(self):
        search = (self.chap_entry.get(), self.sec_entry.get(),
                  self.title_entry.get().lower())

        # Find the elements that satisfy each search box.
        full_matches, found_match = set(ATA_CODES), False
        for i, term in enumerate(search):
            if not term:
                continue
            matches = set(entry for entry in ATA_CODES
                          if term.lower() in entry[i].lower())

            if matches:
                full_matches &= matches
                found_match = True

        if found_match:

            headings = {(entry[0], '', ATA_CODES_[entry[0]][''])
                        for entry in full_matches}  # Always incl. chap hdgs.
            full_matches = sorted(full_matches | headings)
            match_txt = []
            for entry in full_matches:
                if entry[1] == '':
                    match_txt.append(f"\nCHAPTER {entry[0]} - {entry[2]}")
                else:
                    match_txt.append(f"{entry[0]}-{entry[1]} - {entry[2]}")
            match_txt = '\n'.join(match_txt)
        else:
            match_txt = '--- NONE ---'

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.tag_configure('tag-center', justify=tk.CENTER)
        self.result_text.insert(tk.END, match_txt, 'tag-center')
        self.result_text.config(state=tk.DISABLED)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    gui = ATACodeGUI()
    root.mainloop()
