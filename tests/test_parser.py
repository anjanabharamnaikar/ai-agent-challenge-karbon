# tests/test_parser.py

import pandas as pd
import pytest
from custom_parsers.icici_parser import parse

def test_icici_parser():
    """
    Tests the generated ICICI parser against the sample data.
    """
    pdf_path = "data/icici/icici_sample.pdf"
    csv_path = "data/icici/icici_sample.csv"

    expected_df = pd.read_csv(csv_path)
    # Ensure data types are consistent for comparison
    expected_df['Date'] = pd.to_datetime(expected_df['Date'], dayfirst=True)
    
    actual_df = parse(pdf_path)

    pd.testing.assert_frame_equal(actual_df, expected_df)