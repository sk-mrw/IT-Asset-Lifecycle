
# IT Asset Management Dataset - Data Dictionary

## File 1: it_assets_master.csv (Master Asset Database)
- Asset_ID: Unique identifier for each asset
- Serial_Number: Manufacturer serial number
- Manufacturer: Asset manufacturer (Dell, HP, Lenovo, Apple, Microsoft)
- Model: Specific model name
- Asset_Type: Category (Laptop, Desktop, Monitor, Server, Tablet)
- Purchase_Date: Date asset was purchased
- Purchase_Cost: Original purchase cost in USD
- Warranty_End_Date: Date warranty expires (may be null for data quality issues)
- Region: Geographic region where asset is deployed
- Department: Department to which asset is assigned
- Assigned_User: User ID of assigned user (may be null)
- Status: Current asset status (Active, In Storage, In Repair, Retired, Disposed)
- OS_Version: Operating system version (for computers only)
- Encryption_Status: Disk encryption status (Encrypted, Not Encrypted, Unknown)
- Last_Login_Days_Ago: Days since last user login
- Avg_CPU_Usage_Percent: Average CPU utilization (last 30 days)
- Disk_Usage_Percent: Percentage of disk space used
- Has_Failed: Whether asset has experienced failure (Yes/No)
- Age_Years: Asset age in years

## File 2: servicenow_discovery.csv (Discovery Tool Data)
- CI_ID: Configuration Item ID in ServiceNow
- Serial_Number: Serial number (should match master data)
- Discovered_User: User discovered by automated tools
- Discovered_Encryption: Encryption status from discovery
- Last_Discovery_Date: Last successful discovery scan date
- Discovery_Source: Discovery tool used (SCCM, JAMF, Intune)

## File 3: incident_history.csv (Support Ticket History)
- Incident_ID: Unique incident identifier
- Asset_ID: Associated asset ID
- Serial_Number: Asset serial number
- Incident_Date: Date incident was logged
- Issue_Type: Category of issue reported
- Resolution_Days: Days taken to resolve
- Repair_Cost: Cost of repair in USD

## Data Quality Issues (Intentional for Analysis)
- ~15% of assets missing in ServiceNow discovery
- ~12% user assignment discrepancies between systems
- ~10% encryption status discrepancies
- ~8% missing warranty end dates
- ~5% missing assigned users
