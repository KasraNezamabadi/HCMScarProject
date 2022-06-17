import sys  # For system information
import optparse
import xml.parsers.expat
import base64
import struct
import array
import os

# Globals
# The following are the element tags for the XML document
# WAVEFORM_ELEM = "Waveform"
# WAVETYPE_ELEM = "WaveformType"
# RHYTHM_TAG = "Rhythm"
# LEAD_DATA_ELEM = "LeadData"
# LEAD_ID_ELEM = "LeadID"
# WAVEFORM_DATA_ELEM = "WaveFormData"
# SAMPLE_BASE_ELEM = "SampleBase"
# LEAD_ADU_ELEM = "LeadAmplitudeUnitsPerBit"
# LEAD_UNIT_ELEM = "LeadAmplitudeUnits"

INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


###############################################################################
# Classes
###############################################################################
class XmlElementParser:
    """Abstract base class for a XML Parsing State. It contains methods for
    restoring the previous state and for tracking the character data between
    tags."""

    def __init__(self, old_State=None):
        self.__old_State = old_State
        self.__data_Text = ""

    def restoreState(self, context):
        """This method restores the previous state in the XML parser."""
        if self.__old_State:
            context.setState(self.__old_State)

    def clearData(self):
        """This method clears the character data that was collected during parsing"""
        self.__data_Text = ""

    def getData(self):
        """This method returns the character data that was collected during
        parsing and it strips any leading or trailing whitespace"""
        return self.__data_Text.strip()

    def start_element(self, name, attrs, context):
        # print("""abstract method, called at the start of an XML element""")
        sys.exit(0)

    def end_element(self, name, context):
        # print("""abstract method, called at the end of an XML element""")
        sys.exit(0)

    def char_data(self, data, context):
        """This method accumulates any character data"""
        self.__data_Text = self.__data_Text + data


class IdleParser(XmlElementParser):
    """State for handling the Idle condition"""

    def __init__(self):
        XmlElementParser.__init__(self)

    def start_element(self, name, attrs, context):
        if name == WaveformElementParser.Tag:
            context.setState(WaveformElementParser(self))

    def end_element(self, name, context):
        self.clearData()


class WaveformElementParser(XmlElementParser):
    """State for handling the Waveform element"""

    Tag = "Waveform"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()
        if name == WaveformTypeElementParser.Tag:
            context.setState(WaveformTypeElementParser(self))
        elif name == LeadDataElementParser.Tag:
            context.setState(LeadDataElementParser(self))
        elif name == SampleBaseElementParser.Tag:
            context.setState(SampleBaseElementParser(self))

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)


class SampleBaseElementParser(XmlElementParser):
    """State for handling the SampleBase element"""

    Tag = "SampleBase"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            if context.found_Rhythm:
                context.setSampleBase(self.getData())
                # print("Sampling rate for rhythm is %s sps..." % (context.sample_Rate))


class LeadUnitsPerBitElementParser(XmlElementParser):
    """State for handling the LeadAmplitudeUnitsPerBit element"""

    Tag = "LeadAmplitudeUnitsPerBit"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            context.setAdu(float(self.getData().strip()))


class LeadUnitsElementParser(XmlElementParser):
    """State for handling the LeadAmplitudeUnits element"""

    Tag = "LeadAmplitudeUnits"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            context.setUnits(self.getData().strip())


class WaveformTypeElementParser(XmlElementParser):
    """State for handling the WaveformType element"""

    Tag = "WaveformType"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            if 'Rhythm' in self.getData():
                context.setRhythmFound(1)
            else:
                context.setRhythmFound(0)


class LeadDataElementParser(XmlElementParser):
    """State for handling the LeadData element"""

    Tag = "LeadData"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()
        if name == LeadIdElementParser.Tag:
            context.setState(LeadIdElementParser(self))
        if name == WaveformDataElementParser.Tag:
            context.setState(WaveformDataElementParser(self))
        if name == LeadUnitsPerBitElementParser.Tag:
            context.setState(LeadUnitsPerBitElementParser(self))
        if name == LeadUnitsElementParser.Tag:
            context.setState(LeadUnitsElementParser(self))

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)


class LeadIdElementParser(XmlElementParser):
    """State for handling the LeadID element"""

    Tag = "LeadID"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            if context.found_Rhythm:
                #sys.stdout.write("   Lead %2s found..." % self.getData())
                context.addLeadId(self.getData())


class WaveformDataElementParser(XmlElementParser):
    """State for handling the WaveformData element"""

    Tag = "WaveFormData"

    def __init__(self, old_State):
        XmlElementParser.__init__(self, old_State)

    def start_element(self, name, attrs, context):
        self.clearData()

    def end_element(self, name, context):
        if name == self.Tag:
            self.restoreState(context)
            if context.found_Rhythm:
                # print("   Adding data for lead %2s." % context.lead_Id)
                context.addWaveformData(self.getData())


class MuseXmlParser:
    """This class is the parsing context in the object-oriented State pattern."""

    def __init__(self):
        self.ecg_Data = dict()
        self.ecg_Leads = list()
        self.__state = IdleParser()
        self.found_Rhythm = 0
        self.sample_Rate = 0
        self.adu_Gain = 1
        self.units = ""

    def setState(self, s):
        self.__state = s

    def getState(self):
        return self.__state

    def setSampleBase(self, text):
        if self.found_Rhythm:
            self.sample_Rate = int(text)

    def setAdu(self, gain):
        self.adu_Gain = gain

    def setUnits(self, units):
        self.units = units

    def setRhythmFound(self, v):
        self.found_Rhythm = v

    def addLeadId(self, id):
        self.lead_Id = id

    def addWaveformData(self, text):
        self.ecg_Data[self.lead_Id] = base64.b64decode(text)
        self.ecg_Leads.append(self.lead_Id)

    def start_element(self, name, attrs):
        """This function trackes the start elements found in the XML file with a
        simple state machine"""
        self.__state.start_element(name, attrs, self)

    def end_element(self, name):
        self.__state.end_element(name, self)

    def char_data(self, data):
        self.__state.char_data(data, self)

    def makeZcg(self):
        """This function converts the data read from the XML file into a ZCG buffer
        suitable for storage in binary format."""
        # All of the leads should have the same number of samples
        if len(self.ecg_Leads) == 0:
            raise Exception('*** No Rhythm Found. Skipping')
            #sys.exit(-1)
        n = len(self.ecg_Data[self.ecg_Leads[0]])
        # We have 2 bytes per ECG sample, so make our buffer size n * DATAMUX
        self.zcg = array.array('d')
        # Verify that all of the independent leads are accounted for...
        for lead in INDEPENDENT_LEADS:
            if lead not in self.ecg_Leads:
                # print("Error! The XML file is missing data for lead ", lead)
                sys.exit(-1)

        # Append the data into our huge ZCG buffer in the correct order
        for t in range(0, n, 2):
            for lead in self.ecg_Leads:
                sample = struct.unpack("h", bytes([self.ecg_Data[lead][t], self.ecg_Data[lead][t + 1]]))
                # sample = struct.unpack("h", self.ecg_Data[lead][t] + self.ecg_Data[lead][t + 1])
                self.zcg.append(sample[0])

    def writeCSV(self, file_Name):
        """This function writes the ZCG buffer to a CSV file. All 12 or 15 leads
        are generated."""
        std_Leads = set(INDEPENDENT_LEADS)
        header = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
        extra_Leads = std_Leads.symmetric_difference(set(self.ecg_Leads))
        # # print "EXTRA LEADS: ", extra_Leads

        fd = open(file_Name, 'wt')
        if fd:
            # write the header information
            for lead in header:
                fd.write("%s, " % lead)
            # write any extra leads
            for lead in self.ecg_Leads:
                if lead in extra_Leads:
                    fd.write("%s, " % lead)
            fd.write("\n")

            samples = dict()

            for i in range(0, len(self.zcg), len(self.ecg_Leads)):
                # The values in the ZCG buffer are stored in the same order
                # as the ecg leads are themselves...
                k = 0
                for lead in self.ecg_Leads:
                    samples[lead] = self.zcg[i + k]
                    k = k + 1
                # Output each sample, calculated and uncalcuated
                fd.write("%d, " % int(samples["I"] * self.adu_Gain))
                fd.write("%d, " % int(samples["II"] * self.adu_Gain))
                # II - I = III
                fd.write("%d, " % int((samples["II"] - samples["I"]) * self.adu_Gain))
                # aVR = -(I + II)/2
                fd.write("%d, " % int((-(samples["I"] + samples["II"]) / 2) * self.adu_Gain))
                # aVL = I - II/2
                fd.write("%d, " % int((samples["I"] - samples["II"] / 2) * self.adu_Gain))
                # aVF = II - I/2
                fd.write("%d, " % int((samples["II"] - samples["I"] / 2) * self.adu_Gain))
                # output the precordial leads
                fd.write("%d, " % int(samples["V1"] * self.adu_Gain))
                fd.write("%d, " % int(samples["V2"] * self.adu_Gain))
                fd.write("%d, " % int(samples["V3"] * self.adu_Gain))
                fd.write("%d, " % int(samples["V4"] * self.adu_Gain))
                fd.write("%d, " % int(samples["V5"] * self.adu_Gain))
                fd.write("%d, " % int(samples["V6"] * self.adu_Gain))
                # output any extra leads
                for lead in self.ecg_Leads:
                    if lead in extra_Leads:
                        fd.write("%d, " % int(samples[lead] * self.adu_Gain))
                fd.write("\n")
        # print("\nCSV file (\"%s\") is generated, with %d columns of ECG signals" % (file_Name, len(header) + len(extra_Leads)))
        # print("ECG sampling rate is %d Hz." % self.sample_Rate)
        # print("ECG stored in units of %s." % self.units)


g_Parser = None

def start_element(name, attrs):
    g_Parser.start_element(name, attrs)


def end_element(name):
    g_Parser.end_element(name)


def char_data(data):
    g_Parser.char_data(data)


if __name__ == '__main__':

    g_Parser = MuseXmlParser()

    p = xml.parsers.expat.ParserCreate()

    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = char_data

    filename = 'sample.xml'
    p.ParseFile(open(filename, 'rb'))
    g_Parser.makeZcg()
    g_Parser.writeCSV("result.csv")

