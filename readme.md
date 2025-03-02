# AUTOML Zones Extraction Scripts and Guide

Step-by-Step Explanation
Data Parsing: Read gene and transcript data from files, extracting relevant genomic coordinates and nucleotide sequences.

Transition Extraction:

- EI (Exon → Intron): Validate 'gt' splice site, extract 5 nucleotides left and 7 right of intron start.

- IE (Intron → Exon): Validate 'ag' splice site, extract 100 nucleotides left and 5 right of exon start.

- ZE (Intergenic → First Exon): Extract 500 nucleotides left and 50 right of the first exon's start.

- EZ (Last Exon → Intergenic): Extract 50 nucleotides left and 500 right of the last exon's end.

- False Examples: Generate random nucleotide sequences for each transition type when valid extractions aren't possible.

## Data Storage: Save results into CSV files with each nucleotide in separate columns.

### Solution Code

data_ensembl/
    (contains raw data files)
data/
    data_ei.csv
    data_ie.csv
    data_ze.csv
    data_ez.csv
main_extractor.py

#### Data

Gene:
([GEN_ID],[Cord_inicio_gen],[Cord_final_gen],[string_nucleotides],[chromosome_number],[Cord_global_inicio_gen],[Cord_global_final_gen],strand)
example:
([ENSG00000157005.4],[1000],[2482],[...],[3],[187667913],[187671395],false)

Transcript Information Lines:
([transcript_exon1_start, transcript_exon1_end],[transcript_exon2_start, transcript_exon2_end],...,[transcript_exonN_start, transcript_exonN_end],[transcript_count])

Example:
([1000,1240],[2117,2482],[1])

Detailed Extraction Process
The extraction logic implemented in init.ipynb performs the following steps for each gene that contains transcript data:

1. Exon → Intron (EI)
Locate Transition:
Identify the end position of the first exon (e.g., 1240).

Determine Intron Start:
The intron is assumed to start at exon_end + 1 (i.e., 1241).

Validation:
Confirm that the intron starts with the nucleotides "gt", which should be located at positions 1241–1242.

Extraction:

Extract 5 characters immediately to the left of the intron start.
Extract 5 characters immediately to the right of the intron start.
Concatenate these substrings to form a 12-character sequence.
Storage:
Save the result into data_ei.csv. Each character of the sequence is stored in its own column (B1, B2, …, B12).

2. Intron → Exon (IE)
Locate Transition:
Identify the start position of the second exon (e.g., 2117).

Determine Intron End:
The intron is assumed to end at exon_start - 1 (i.e., 2116).

Validation:
Confirm that the intron ends with "ag", found at positions 2115–2116.

Extraction:

Extract 100 characters immediately to the left of the exon start.
Extract 5 characters immediately to the right of the exon start.
Concatenate these to form a 105-character sequence.
Storage:
Save the result into data_ie.csv, with each character in a separate column (B1 to B105).

3. Intergenic Zone → First Exon (ZE)
Locate Transition:
Use the start position of the first exon (e.g., 1000).

Extraction:

Extract 500 characters immediately to the left of the first exon.
Extract 50 characters immediately to the right of the first exon.
Concatenate these to form a 550-character sequence.
Storage:
Save the result into data_ze.csv, with each character occupying its own column (B1 to B550).

4. Last Exon → Intergenic Zone (EZ)
Locate Transition:
Use the end position of the last exon (e.g., 2482).

Extraction:

Extract 50 characters immediately to the left of the last exon.
Extract 500 characters immediately to the right of the last exon.
Concatenate these to form a 550-character sequence.
Storage:
Save the result into data_ez.csv, with each character in its respective column (B1 to B550).

Files and Outputs
After running the init.ipynb notebook, the following CSV files are generated in the designated data folder:

data_ei.csv: Contains the EI transition sequences.
data_ie.csv: Contains the IE transition sequences.
data_ze.csv: Contains the ZE transition sequences.
data_ez.csv: Contains the EZ transition sequences.
Each CSV file includes metadata (such as gene ID, chromosome number, and genomic coordinates) along with the extracted transition sequence distributed across multiple columns.
