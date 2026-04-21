import re
from typing import Pattern

# Constant regex pattern for a UniProt accession
UNIPROT_ACCESSION_PATTERN: Pattern[str] = re.compile(
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
)
