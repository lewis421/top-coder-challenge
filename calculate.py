# calculate.py
import sys
from reimbursement_model_knn import Model

# Initialize a model instance
reimbursement_engine = Model()

# Load the pre-trained artifacts from the file
# This happens once when the script starts
if not reimbursement_engine.load_artifacts():
    sys.exit(1)

# Get inputs from command-line arguments
try:
    days_in = int(sys.argv[1])
    miles_in = int(float(sys.argv[2]))
    receipts_in = float(sys.argv[3])
except (ValueError, IndexError):
    print(f"Usage: python {sys.argv[0]} <days> <miles> <receipts>", file=sys.stderr)
    sys.exit(1)

# Call the calculation method
final_amount = reimbursement_engine.calculate_reimbursement(days_in, miles_in, receipts_in)

# Output the single numeric result
print(f"{final_amount:.2f}")