# -*- coding: utf-8 -*-
"""
Filename: retrieve_usgs.py
Author: Lindsey Lu
Created: 2026-05-26
Version: 1.0
Description: Retrieves annual maximum gage height from USGS gage USGS-01554000.
             Plots the annual maximum gage height over time to determine coefficients for b1 and b2.
"""

import dataretrieval.waterdata as waterdata
import pandas as pd

site_id = "01554000"

print(f"Fetching annual peak data for site: {site_id}...")

# Use the specific peaks service instead of get_info
peak_data, metadata = waterdata.get_peaks(monitoring_location_id=site_id,
                                          parameter_code="00060")

if not peak_data.empty:
    # Clean and filter the results
    columns_to_keep = [col for col in ['site_no', 'peak_dt', 'gage_ht', 'peak_va'] if col in peak_data.columns]
    df_peaks = peak_data[columns_to_keep].copy()
    
    # Format dates and extract year
    df_peaks['peak_dt'] = pd.to_datetime(df_peaks['peak_dt'])
    df_peaks['year'] = df_peaks['peak_dt'].dt.year
    
    # Drop records missing gage height (common in older historical logs)
    df_peaks = df_peaks.dropna(subset=['gage_ht'])
    df_peaks = df_peaks.sort_values('peak_dt').reset_index(drop=True)
    
    print(df_peaks.head())
    df_peaks.to_csv(f"usgs_{site_id}_annual_peak_discharge.csv", index=False)
else:
    print("No peak data found.")