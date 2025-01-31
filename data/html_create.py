import os

# Base URL structure
BASE_URL = "http://archive.ambermd.org/{year}{month:02}/{number:04}.html"

# Output text file for saving URLs
OUTPUT_FILE = "amber_archive_urls.txt"

# Define the year and month ranges (set specific values or ranges)
years = range(2024, 2025)  # Single year or range of years
months = range(12,13)  # Specific month or list of months (e.g., range(1, 13) for all months)
numbers = range(0,21)  # 0000 to 9999

# Open the output file in write mode
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    # Loop through each combination of year, month, and file number
    for year in years:
        for month in months:
            for number in numbers:
                # Format the URL
                url = BASE_URL.format(year=year, month=month, number=number)

                # Write the URL to the file
                file.write(url + "\n")

print(f"URLs have been written to {OUTPUT_FILE}")
