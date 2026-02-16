import argparse
import csv
from pathlib import Path
import cobra 
import pandas as pd
import numpy as np

# ==== STEP 1: Create supplementary mapping CSV ====

def create_supplementary_csv(model: cobra.Model, output_dir: Path) -> Path:
    """
    Create AIDA compound → GEM exchange reaction mapping CSV.
    This structure supports 1-to-many mappings (salts) and stoichiometry.
    """
    # Mapping: { AIDA_Name: { GEM_Rxn: Multiplier } }
    mapping_rules = {
        # --- Single Matches ---

        # --- Carbon Sources ---
        "Glucose (mM)": {"EX_glc__D_e": 1.0},

        # --- Amino Acids ---
        "Alanine (mM)": {"EX_ala__L_e": 1.0},
        "Arginine/HCl (mM)": {"EX_arg__L_e": 1.0},
        "Asparagine/H2O (mM)": {"EX_asn__L_e": 1.0},
        "AsparticAcid (mM)": {"EX_asp__L_e": 1.0},
        "GlutamicAcid/HCl (mM)": {"EX_glu__L_e": 1.0},
        "Glutamine (mM)": {"EX_gln__L_e": 1.0},
        "Glycine (mM)": {"EX_gly_e": 1.0},
        "Histidine/HCl/H2O (mM)": {"EX_his__L_e": 1.0},
        "Isoleucine (mM)": {"EX_ile__L_e": 1.0},
        "Leucine (mM)": {"EX_leu__L_e": 1.0},
        "Lysine/HCl (mM)": {"EX_lys__L_e": 1.0},
        "Methionine (mM)": {"EX_met__L_e": 1.0},
        "Phenylalanine (mM)": {"EX_phe__L_e": 1.0},
        "Proline (mM)": {"EX_pro__L_e": 1.0},
        "Serine (mM)": {"EX_ser__L_e": 1.0},
        "Threonine (mM)": {"EX_thr__L_e": 1.0},
        "Tryptophan (mM)": {"EX_trp__L_e": 1.0},
        "Tyrosine (mM)": {"EX_tyr__L_e": 1.0},
        "Valine (mM)": {"EX_val__L_e": 1.0},
        "Thiamine/HCl (mM)": {"EX_thm_e": 1.0},
        "Pyridoxine (mM)": {"EX_pydxn_e": 1.0},
        # or "Cystine/HCl/H2O (mM)": {"EX_cys__L_e": 2.0},  

        # --- Ambiguous / Salt Matches ---

        "Cystine/HCl/H2O (mM)": {"EX_cys__L_e": 2.0, "EX_cl_e": 2.0},  # Cystine is 2x Cysteine + 2x Cl

        # --- Nitrogen, Phosphorus, Sulfur ---
        "(NH4)2SO4 (mM)": {"EX_nh4_e": 2.0, "EX_so4_e": 1.0},
        "NH4Cl (mM)": {"EX_nh4_e": 1.0, "EX_cl_e": 1.0},
        
        "K2HPO4 (mM)": {"EX_pi_e": 1.0, "EX_k_e": 2.0},
        "KH2PO4 (mM)": {"EX_pi_e": 1.0, "EX_k_e": 1.0},
        "Na2HPO4 (mM)": {"EX_pi_e": 1.0, "EX_na1_e": 2.0},

        # --- Magnesium and Calcium ---
        "MgCl2/6H2O (mM)": {"EX_mg2_e": 1.0, "EX_cl_e": 2.0},
        "MgSO4/7H2O (mM)": {"EX_mg2_e": 1.0, "EX_so4_e": 1.0},
        "CaCl2/2H2O (mM)": {"EX_ca2_e": 1.0, "EX_cl_e": 2.0},
        "CaSO4/2H2O (mM)": {"EX_ca2_e": 1.0, "EX_so4_e": 1.0},

        # --- Iron and Trace Metals ---
        "FeSO4/7H2O (mM)": {"EX_fe2_e": 1.0, "EX_so4_e": 1.0},
        "ZuSO4/7H2O (mM)": {"EX_zn2_e": 1.0, "EX_so4_e": 1.0}, # Zinc (typo in Aida)
        "CuSO4/5H2O (mM)": {"EX_cu2_e": 1.0, "EX_so4_e": 1.0},
        "Na2MoO4/2H2O (mM)": {"EX_mobd_e": 1.0, "EX_na1_e": 2.0},
        
        # --- Others ---
        "KCl (mM)": {"EX_k_e": 1.0, "EX_cl_e": 1.0},
        "NaCl (mM)": {"EX_na1_e": 1.0, "EX_cl_e": 1.0},
        "Na3C6H5O7/2H2O (mM)": {"EX_cit_e": 1.0, "EX_na1_e": 3.0}, # Citrate
        "Na2S2O3/5H2O (mM)": {"EX_tsul_e": 1.0, "EX_na1_e": 2.0},  # Thiosulfate
    }

    print(f"Total AIDA compounds in mapping: {len(mapping_rules)}")
    valid_rxns = {r.id for r in model.reactions}
    out_csv = output_dir / "aidaying_compound_mapping.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AIDA_compound", "gem_rxn", "stoichiometry", "match_type"])

        for comp, rxns in mapping_rules.items():
            # Classify based on number of mapped reactions
            match_type = "single_match" if len(rxns) == 1 else "ambiguous"
            
            for rxn, coeff in rxns.items():
                if rxn in valid_rxns:
                    writer.writerow([comp, rxn, coeff, match_type])
                else:
                    writer.writerow([comp, rxn, coeff, "missing_match"])
                    print(f"WARNING: Reaction {rxn} not found in GEM for {comp}")

    return out_csv

# ==== STEP 2: Build mapping matrix ====

def build_mapping_matrix(mapping_csv: Path, media_columns: list) -> pd.DataFrame:
    """Constructs M matrix: Rows=AIDA, Cols=GEM reactions."""
    map_df = pd.read_csv(mapping_csv)
    
    # Identify which AIDA compounds from our mapping actually exist in the CSV
    present_aida_cols = [c for c in map_df["AIDA_compound"].unique() if c in media_columns]
    unique_gem_rxns = map_df["gem_rxn"].unique().tolist()

    # Initialize zero matrix
    M = pd.DataFrame(0.0, index=present_aida_cols, columns=unique_gem_rxns)

    for _, row in map_df.iterrows():
        if row["AIDA_compound"] in M.index:
            M.loc[row["AIDA_compound"], row["gem_rxn"]] = row["stoichiometry"]
    
    return M

# ==== STEP 3: Create media and growth CSV ====

def create_media_growth_csv(aida_dir: Path, mapping_csv: Path, output_dir: Path) -> pd.DataFrame:
    media_csv = aida_dir / "aidaying_medium.csv" 
    growth_csv = aida_dir / "aidaying_growth.csv"

    media_df = pd.read_csv(media_csv)
    growth_df = pd.read_csv(growth_csv)

    # Normalize media column names
    media_df.columns = (
        media_df.columns
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Build and apply mapping matrix
    M = build_mapping_matrix(mapping_csv, media_df.columns.tolist())
    
    # Identify what was NOT renamed/mapped
    not_renamed_cols = [
        col for col in media_df.columns 
        if col not in M.index and col != "Condition ID"
    ]
    
    print(f"Mapped {len(M.index)} AIDA columns into {len(M.columns)} GEM reactions.")
    print(f"{len(not_renamed_cols)} media columns were not mapped.")
    print(f"Columns not renamed: {not_renamed_cols}")

    # Use Matrix Multiplication to transform the data (Handles Summing & Stoichiometry)
    A = media_df[M.index].fillna(0.0)
    gem_values_df = A.dot(M)
    
    # Combine back with Condition ID
    media_gem_df = pd.concat([media_df[["Condition ID"]], gem_values_df], axis=1)
    gem_rxns = M.columns.tolist()

    # Deduplicate by unique media composition
    print(f"Media compositions before dropping duplicates → {media_gem_df.shape[0]}")
    media_gem_df = media_gem_df.drop_duplicates(subset=gem_rxns).reset_index(drop=True)
    print(f"Unique media compositions after dropping duplicates → {media_gem_df.shape[0]}")

    # Get average growth data (r_info filtering)
    print(f"Total growth entries in CSV: {len(growth_df)}")
    print(f"Entries with r_info = 1: {len(growth_df[growth_df['r_info'] == 1])}")
    print(f"Entries with r_info = 2: {len(growth_df[growth_df['r_info'] == 2])}")

    # Only keep r_info = 0
    growth_df = growth_df[growth_df["r_info"] == 0]
    print(f"Growth entries remaining (r_info = 0): {len(growth_df)}")

    avg_growth = (
        growth_df
        .groupby("Condition ID", as_index=False)
        .agg(avg_growth=("r", "mean"))
    )

    # Merge media + growth
    final_df = pd.merge(media_gem_df, avg_growth, on="Condition ID", how="inner")

    # Reorder columns
    final_df = final_df[["Condition ID", "avg_growth"] + gem_rxns]
    
    # Keep only unique condition IDs
    final_df = final_df.drop_duplicates(subset=["Condition ID"])
    final_df.to_csv(output_dir / "aidaying_media_growth_amns.csv", index=False)
    
    print(f"Final unique conditions → {final_df.shape[0]}")
    print(f"Final media data saved to {output_dir / 'aidaying_media_growth_amns.csv'}")

    return final_df

# ==== STEP 4: Match against corrected medium compositions ====

def match_amns_media(amns_dir: Path, output_dir: Path, final_df: pd.DataFrame) -> Path:
    amns_media_csv = amns_dir / "iML1515_medium_corrected.csv"
    if not amns_media_csv.exists():
        print(f"ERROR: Correction file not found: {amns_media_csv}")
        return None

    # Load and normalize AMNS table
    amns_media_df = pd.read_csv(amns_media_csv)

    # Strip whitespace from rxn names (CRITICAL)
    amns_media_df["rxn"] = amns_media_df["rxn"].astype(str).str.strip()

    # Build correction map: {EX_xxx_i -> flux}
    correction_map = dict(
        zip(amns_media_df["rxn"], amns_media_df["value"])
    )

    # Prepare corrected medium dataframe
    corrected_df = final_df.copy()

    # Normalize column names
    corrected_df.columns = [c.strip() for c in corrected_df.columns]

    base_cols = [
        c for c in corrected_df.columns
        if c not in ["Condition ID", "avg_growth"]
    ]

    # Rename columns to add '_i' suffix (for columns that don't already have it)
    num_renamed = 0
    rename_map = {}
    
    for col in base_cols:
        if not col.endswith("_i"):
            suffixed_col = f"{col}_i"
            rename_map[col] = suffixed_col
            num_renamed += 1
    
    if rename_map:
        corrected_df.rename(columns=rename_map, inplace=True)
    
    # Verify columns exist in AMNS reference
    num_matched = 0
    missing_in_amns = set()
    
    # Get updated column names after renaming
    updated_cols = [
        c for c in corrected_df.columns
        if c not in ["Condition ID", "avg_growth"]
    ]
    
    for col in updated_cols:
        if col in correction_map:
            num_matched += 1
        else:
            missing_in_amns.add(col)
            
    # Keep only unique condition IDs
    corrected_df = corrected_df.drop_duplicates(subset=["Condition ID"])
    
    # Save dataframe with condition IDs
    corrected_df.to_csv(output_dir / "aidaying_media_growth.csv", index=False)
    
    # Rename growth column
    corrected_df.rename(columns={"avg_growth": "GR_AVG"}, inplace=True)
    
    # Drop Condition ID column
    corrected_df.drop(columns=["Condition ID"], inplace=True)
    
    # Reorder columns
    corrected_df = corrected_df[updated_cols + ["GR_AVG"]]
    
    # Save output
    corrected_csv = output_dir / "aidaying_media_growth_amns.csv"
    corrected_df.to_csv(corrected_csv, index=False)

    # Logging numbers
    print(f"Columns matched and renamed with AMNS reference: {num_matched}/{len(updated_cols)}")
    print(f"Final media data saved to {corrected_csv}")
    
    if missing_in_amns:
        print(f"WARNING: Columns in data but missing in AMNS reference: {len(missing_in_amns)}")
        print(f"Missing reactions: {missing_in_amns}")
    
    return corrected_csv


# ==== STEP 5: Main execution ====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=Path("data/raw/iML1515.xml"))
    parser.add_argument("--aida_dir", type=Path, default=Path("data/external"))
    parser.add_argument("--amns_dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()
 
    print(f"Loading GEM → {args.model_path}")
    model = cobra.io.read_sbml_model(str(args.model_path))

    mapping_csv = create_supplementary_csv(model, args.output_dir)
    final_df = create_media_growth_csv(args.aida_dir, mapping_csv, args.output_dir)
    corrected_csv = match_amns_media(args.amns_dir, args.output_dir, final_df)

if __name__ == "__main__":
    main()