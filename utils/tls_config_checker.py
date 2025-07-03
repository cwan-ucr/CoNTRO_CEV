import xml.etree.ElementTree as ET

class TLSConfigChecker:
    def __init__(self, net_xml_path):
        self.net_xml_path = net_xml_path
        self.actuated_tls_ids = []

    def detect_actuated_tls(self):
        """
        Scans the .net.xml file and records all actuated TLS IDs.
        """
        try:
            tree = ET.parse(self.net_xml_path)
            root = tree.getroot()

            for tl in root.findall('tlLogic'):
                tls_id = tl.get('id')
                control_type = tl.get('type')

                if control_type and control_type.lower() == 'actuated':
                    self.actuated_tls_ids.append(tls_id)

        except ET.ParseError as e:
            print(f"[TLSConfigChecker] XML parsing error: {e}")
        except FileNotFoundError:
            print(f"[TLSConfigChecker] File not found: {self.net_xml_path}")

        return self.actuated_tls_ids

    def is_actuated_mode(self):
        """
        Returns True if any actuated TLS is found.
        """
        return len(self.actuated_tls_ids) > 0
