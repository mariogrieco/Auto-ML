import random
import re
import pandas as pd

# EIExtractor - Exon → Intron
class EIExtractor:
    def __init__(self):
        self.true_data = []   # Stores true EI transitions
        self.false_data = []  # Stores false EI transitions

    def extract_true(self, gen_id, chromosome, global_start, sequence, exons):
        for i in range(len(exons) - 1):
            exon_end = exons[i][1]
            intron_start = exon_end + 1

            # Check if there are enough characters and the intron starts with 'gt'
            if intron_start + 1 < len(sequence) and sequence[intron_start:intron_start + 2] == "gt":
                # Extract 5 nucleotides to the left and 7 to the right
                left = sequence[max(0, intron_start - 5):intron_start]
                right = sequence[intron_start:intron_start + 7]
                transition_seq = left + right
                self.true_data.append([gen_id, chromosome, global_start, exon_end, *list(transition_seq)])

    def extract_false_random(self, gen_id, chromosome, global_start):
        nucleotides = "acgt"
        false_chars = [random.choice(nucleotides) for _ in range(12)]
        false_chars[5] = 'g'
        false_chars[6] = 't'
        false_seq = "".join(false_chars)
        self.false_data.append([gen_id, chromosome, global_start, None, *list(false_seq)])

    def get_data(self):
        return self.true_data, self.false_data


# EZExtractor - Last Exon → Intergenic Zone
class EZExtractor:
    def __init__(self):
        self.true_data = []
        self.false_data = []

    def extract_true(self, gen_id, chromosome, global_start, sequence, exons):
        exon_end = exons[-1][1]
        left = sequence[max(0, exon_end - 50):exon_end]
        right = sequence[exon_end:exon_end + 500]
        transition_seq = left + right
        self.true_data.append([gen_id, chromosome, global_start, exon_end, *list(transition_seq)])

    def extract_false_random(self, gen_id, chromosome, global_start):
        nucleotides = "acgt"
        false_seq = "".join(random.choice(nucleotides) for _ in range(550))
        self.false_data.append([gen_id, chromosome, global_start, None, *list(false_seq)])

    def get_data(self):
        return self.true_data, self.false_data


# IEExtractor - Intron → Exon
class IEExtractor:
    def __init__(self):
        self.true_data = []
        self.false_data = []

    def extract_true(self, gen_id, chromosome, global_start, sequence, exons):
        for i in range(len(exons) - 1):
            exon_start = exons[i + 1][0]
            intron_end = exon_start - 1

            # Check if there are enough characters and the intron ends with 'ag'
            if intron_end - 1 >= 0 and sequence[intron_end - 1:intron_end + 1] == "ag":
                left = sequence[max(0, intron_end - 100):intron_end]
                right = sequence[intron_end:intron_end + 5]
                transition_seq = left + right  # 100 + 5 = 105 characters
                self.true_data.append([gen_id, chromosome, global_start, exon_start, *list(transition_seq)])

    def extract_false_random(self, gen_id, chromosome, global_start):
        nucleotides = "acgt"
        false_seq = "".join(random.choice(nucleotides) for _ in range(105))
        self.false_data.append([gen_id, chromosome, global_start, None, *list(false_seq)])

    def get_data(self):
        return self.true_data, self.false_data


# ZEExtractor - Intergenic Zone → First Exon
class ZEExtractor:
    def __init__(self):
        self.true_data = []
        self.false_data = []

    def extract_true(self, gen_id, chromosome, global_start, sequence, exons):
        exon_start = exons[0][0]
        left = sequence[max(0, exon_start - 500):exon_start]
        right = sequence[exon_start:exon_start + 50]
        transition_seq = left + right  # 500 + 50 = 550 characters
        self.true_data.append([gen_id, chromosome, global_start, exon_start, *list(transition_seq)])

    def extract_false_random(self, gen_id, chromosome, global_start):
        nucleotides = "acgt"
        false_seq = "".join(random.choice(nucleotides) for _ in range(550))
        self.false_data.append([gen_id, chromosome, global_start, None, *list(false_seq)])

    def get_data(self):
        return self.true_data, self.false_data


# Main Extraction Class
class Extraction:
    def __init__(self, file_path, output_path="./data"):
        self.file_path = file_path
        self.output_path = output_path

        # Read file contents
        with open(self.file_path, "r") as f:
            self.lines = f.readlines()

        # Regex pattern to identify transcript lines
        self.transcript_regex = re.compile(r"^\(\[(\d+,\d+)](,\[(\d+,\d+)])*,\[(\d+)]\)$")

        # Instantiate each zone extractor
        self.ei_extractor = EIExtractor()
        self.ie_extractor = IEExtractor()
        self.ze_extractor = ZEExtractor()
        self.ez_extractor = EZExtractor()

    def process_file(self):
        index = 0
        while index < len(self.lines):
            line = self.lines[index].strip()
            if line.startswith("("):
                # Extract gene information using regex
                match = re.match(
                    r"\(\[(.*?)],\[(\d+)],\[(\d+)],\[(.*?)],\[(\d+)],\[(\d+)],\[(\d+)],(true|false)\)",
                    line
                )
                if match:
                    gen_id, start, end, sequence, chromosome, global_start, global_end, strand = match.groups()
                    start, end, chromosome, global_start, global_end = map(int, [start, end, chromosome, global_start, global_end])

                    # Accumulate transcript lines (exon details)
                    exons_list = []
                    while index + 1 < len(self.lines) and self.transcript_regex.match(self.lines[index + 1].strip()):
                        index += 1
                        trans_line = self.lines[index].strip()
                        exon_matches = re.findall(r"\[(\d+),(\d+)]", trans_line)
                        exons = [(int(s), int(e)) for s, e in exon_matches]
                        exons_list.append(exons)

                    # For each set of exons, delegate extraction to each extractor
                    for exons in exons_list:
                        # EI extraction
                        self.ei_extractor.extract_true(gen_id, chromosome, global_start, sequence, exons)
                        self.ei_extractor.extract_false_random(gen_id, chromosome, global_start)

                        # IE extraction
                        self.ie_extractor.extract_true(gen_id, chromosome, global_start, sequence, exons)
                        self.ie_extractor.extract_false_random(gen_id, chromosome, global_start)

                        # ZE extraction
                        self.ze_extractor.extract_true(gen_id, chromosome, global_start, sequence, exons)
                        self.ze_extractor.extract_false_random(gen_id, chromosome, global_start)

                        # EZ extraction
                        self.ez_extractor.extract_true(gen_id, chromosome, global_start, sequence, exons)
                        self.ez_extractor.extract_false_random(gen_id, chromosome, global_start)
            index += 1

    def save_to_csv(self):
        # EI
        ei_true, ei_negative = self.ei_extractor.get_data()
        pd.DataFrame(ei_true).to_csv(
            f"{self.output_path}/data_ei.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_End"] + [f"B{i + 1}" for i in range(12)]
        )
        pd.DataFrame(ei_negative).to_csv(
            f"{self.output_path}/data_ei_random.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_End"] + [f"B{i + 1}" for i in range(12)]
        )

        # IE
        ie_true, ie_negative = self.ie_extractor.get_data()
        pd.DataFrame(ie_true).to_csv(
            f"{self.output_path}/data_ie.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_Start"] + [f"B{i + 1}" for i in range(105)]
        )
        pd.DataFrame(ie_negative).to_csv(
            f"{self.output_path}/data_ie_random.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_Start"] + [f"B{i + 1}" for i in range(105)]
        )

        # ZE
        ze_true, ze_negative = self.ze_extractor.get_data()
        pd.DataFrame(ze_true).to_csv(
            f"{self.output_path}/data_ze.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_Start"] + [f"B{i + 1}" for i in range(550)]
        )
        pd.DataFrame(ze_negative).to_csv(
            f"{self.output_path}/data_ze_random.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_Start"] + [f"B{i + 1}" for i in range(550)]
        )

        # EZ
        ez_true, ez_negative = self.ez_extractor.get_data()
        pd.DataFrame(ez_true).to_csv(
            f"{self.output_path}/data_ez.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_End"] + [f"B{i + 1}" for i in range(550)]
        )
        pd.DataFrame(ez_negative).to_csv(
            f"{self.output_path}/data_ez_random.csv", index=False,
            header=["GEN_ID", "Chromosome", "Global_Start", "Exon_End"] + [f"B{i + 1}" for i in range(550)]
        )


# Usage example
if __name__ == "__main__":
    extractor = Extraction("./data_ensembl/21-1-46709983.txt")
    extractor.process_file()
    extractor.save_to_csv()
