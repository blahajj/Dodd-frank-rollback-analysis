import os
import pandas as pd
import concurrent.futures
import csv

def find_series_in_files(series_codes, data_dir="data"):
    """
    Find which CSV file contains each of the given UBPR series codes.
    
    Parameters:
    -----------
    series_codes : list
        List of UBPR series codes to find
    data_dir : str
        Directory containing the FFIEC CSV files
    
    Returns:
    --------
    dict : Dictionary mapping series codes to their file locations
    """
    results = {code: "Not found" for code in series_codes}
    remaining_codes = set(series_codes)
    
    # Get all the CSV files in the data directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Searching through {len(all_files)} CSV files in {data_dir}...")
    
    def check_file(filename):
        """Check if the given file contains any of the series codes"""
        local_results = {}
        file_path = os.path.join(data_dir, filename)
        
        print(f"Checking {filename}...")
        
        try:
            # Read just the header rows to get column names
            header_df = pd.read_csv(file_path, nrows=0)
            columns = set(header_df.columns)
            
            # Check each remaining code
            for code in remaining_codes:
                if code in columns:
                    local_results[code] = filename
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        
        return local_results
    
    # Use parallel processing to speed up the search
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(check_file, filename): filename for filename in all_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                local_results = future.result()
                results.update(local_results)
                
                # Remove found codes from the remaining set
                for code in local_results:
                    if code in remaining_codes:
                        remaining_codes.remove(code)
                
                # If all codes are found, we can stop
                if not remaining_codes:
                    break
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def save_results_to_csv(results, output_file="ubpr_series_locations.csv"):
    """Save the results to a CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UBPR Code', 'File Location'])
        for code, location in results.items():
            writer.writerow([code, location])
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # List of UBPR series codes to find
    series_codes = [
        "UBPRE630", "UBPR7204", "UBPR7206", "UBPR7205", 
        "UBPRE595", "UBPRK447", "UBPRE591", "UBPRE589", 
        "UBPRE415", "UBPRE424", "UBPRE423", "UBPRE005", 
        "UBPRE088", "UBPRE013", "UBPRE018", "UBPRE004"
    ]
    
    # Find the files containing these series
    results = find_series_in_files(series_codes)
    
    # Display the results
    print("\nResults:")
    print("-" * 80)
    for code, location in results.items():
        print(f"{code}: {location}")
    print("-" * 80)
    
    # Save results to CSV
    # save_results_to_csv(results)
    
    # Create a CSV file for the auto_DiD format
    with open("ubpr_metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'name', 'form'])
        
        # Map series codes to their names and forms (files without the 'Combined.csv' part)
        names = {
            "UBPRE630": "Return on Equity",
            "UBPR7204": "Tier 1 Leverage Capital Ratio",
            "UBPR7206": "Tier 1 Risk-Based Capital Ratio",
            "UBPR7205": "Total Risk-Based Capital Ratio",
            "UBPRE595": "Brokered Deposits to Total Deposits",
            "UBPRK447": "Net Non-Core Funding Dependence",
            "UBPRE591": "Core Deposits as % of Total Assets",
            "UBPRE589": "Short Term Investments as % of Total Assets",
            "UBPRE415": "1-4 Family Residential Loans",
            "UBPRE424": "Loans to Individuals",
            "UBPRE423": "Commercial & Industrial Loans",
            "UBPRE005": "Non-Interest Expense / Average Assets",
            "UBPRE088": "Efficiency Ratio",
            "UBPRE013": "Return on Average Assets",
            "UBPRE018": "Net Interest Margin",
            "UBPRE004": "Noninterest Income / Average Assets"
        }
        
        for code, location in results.items():
            if location != "Not found":
                form = location.replace(" Combined.csv", "")
                name = names.get(code, code)
                writer.writerow([code, name, form])
    
    print(f"Auto-DiD format CSV saved to ubpr_metrics.csv")