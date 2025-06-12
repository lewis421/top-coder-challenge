#!/bin/bash

# This script serves as the primary execution wrapper for the reimbursement calculation.
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

# Execute the python inference script.
python3 calculate.py "$1" "$2" "$3"