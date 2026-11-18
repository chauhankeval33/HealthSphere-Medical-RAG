
from src.medical_summary import generate_medical_summary

sample_report = """
Complete Blood Count (CBC)

Hemoglobin: 12.8 g/dL
Reference Range: 12.0 - 16.0 g/dL

WBC: 7.4 x 10^3/uL
Reference Range: 4.0 - 11.0 x 10^3/uL

Platelets: 250 x 10^3/uL
Reference Range: 150 - 450 x 10^3/uL

This is a fictional sample report for software testing.
"""

print("Generating medical summary using Gemini...\n")

summary = generate_medical_summary(sample_report)

print("\n========== MEDICAL SUMMARY ==========\n")
print(summary)

print("\nGemini medical summary test completed!")
