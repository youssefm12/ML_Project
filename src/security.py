"""
=============================================================================
Phase 3.2: Data Security, Privacy, & Governance
=============================================================================
This module provides standardized utilities to anonymize Personally Identifiable 
Information (PII) before it is committed to data stores or exposed to external ML 
pipelines. Additionally, it contains Pydantic schemas establishing a rigid API 
boundary that prevents malicious or unexpected payload injections.
"""

import hashlib
import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("security")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA ANONYMIZATION & PRIVACY LOGIC                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def anonymize_customer_id(df: pd.DataFrame, col_name: str = "CustomerID") -> pd.DataFrame:
    """Hash CustomerID using computationally expensive SHA-256 cryptography."""
    df_out = df.copy()
    if col_name in df_out.columns:
        df_out[col_name] = df_out[col_name].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else x
        )
        logger.info(f"Anonymized column: {col_name} via SHA-256")
    return df_out


def mask_ip_addresses(df: pd.DataFrame, col_name: str = "LastLoginIP") -> pd.DataFrame:
    """Mask extreme ends of IP addresses to obscure precise geolocation footprints."""
    df_out = df.copy()
    if col_name in df_out.columns:
        def _mask_ip(ip: str):
            if pd.isna(ip): return ip
            parts = str(ip).split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.***"
            elif ":" in str(ip): # Handle IPv6 naively
                return "*** IPv6 Masked ***"
            return "***"
        
        df_out[col_name] = df_out[col_name].apply(_mask_ip)
        logger.info(f"Masked precise locators in: {col_name}")
    return df_out


def remove_pii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline block ensuring strictly sensitive identifiers 
    never reach modeling storage boundaries.
    """
    logger.info("--- Starting PII Stripping Process ---")
    df_out = df.copy()
    
    # Irreversibly obfuscate logical keys
    df_out = anonymize_customer_id(df_out)
    df_out = mask_ip_addresses(df_out)
    
    # Drop absolute clear-text identifiers (if they ever exist in our pipeline)
    clear_text_violators = ["Email", "PhoneNumber", "FirstName", "LastName", "PhysicalAddress"]
    cols_to_drop = [c for c in clear_text_violators if c in df_out.columns]
    
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)
        logger.info(f"Hard-dropped strictly prohibited clear-text columns: {cols_to_drop}")
        
    return df_out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. INFERENCING BOUNDARY PROTECTION (PYDANTIC SCHEMAS)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CustomerInferencePayload(BaseModel):
    """
    Strict boundary schema to govern incoming JSON prediction requests 
    during Phase 4 API modeling. Drops bad types and enforces ML-ready logic limits.
    """
    # Behavioral
    Frequency: int = Field(..., ge=0, description="Total number of valid historical transactions")
    MonetaryTotal: float = Field(..., ge=0.0, description="Total amount of lifetime capital spent")
    TotalQuantity: int = Field(..., description="Total products purchased natively")
    ReturnRatio: float = Field(0.0, ge=0.0, le=1.0, description="Percentage of returned purchases")
    
    # Statistical
    MonetaryAvg: float = Field(0.0, description="Average ticket price per invoice")
    MonetaryStd: float = Field(0.0, ge=0.0, description="Standard Deviation of ticket price")
    MonetaryMin: float = Field(0.0)
    MonetaryMax: float = Field(0.0)
    
    # Product Diversity
    UniqueProducts: int = Field(0, description="Unique SKUs purchased")
    UniqueDescriptions: int = Field(0, description="Unique semantic descriptions")
    AvgProductsPerTransaction: float = Field(0.0, ge=0.0)
    
    # Transaction Math
    NegativeQuantityCount: int = Field(0, ge=0)
    ZeroPriceCount: int = Field(0, ge=0)
    CancelledTransactions: int = Field(0, ge=0)
    TotalTransactions: int = Field(0, ge=0)
    UniqueInvoices: int = Field(0, ge=0)
    AvgLinesPerInvoice: float = Field(0.0, ge=0.0)
    
    # Demographic / Support (Can be NaN initially but typed mathematically)
    Age: Optional[int] = Field(None, ge=18, le=120)
    SupportTicketsCount: Optional[int] = Field(0, ge=0)
    SatisfactionScore: Optional[int] = Field(None, ge=0, le=5)

    class Config:
        # Ignore random external form fields not recognized by this strict inference boundary.
        extra = "ignore"
        
    @validator("Age")
    def enforce_adult_minimum(cls, v):
        """Domain restriction logic intercepting anomalies."""
        if v is not None and v < 18:
            raise ValueError(f"Age {v} violates Adult customer structural boundary.")
        return v
        
    @validator("ReturnRatio")
    def enforce_ratio_bound(cls, v):
        if v > 1.0 or v < 0.0:
            raise ValueError(f"Feature 'ReturnRatio' logically bounded to [0.0, 1.0], got {v}")
        return v


def validate_inference_payload(json_dict: Dict[str, Any]) -> CustomerInferencePayload:
    """
    Validates dictionary against the Inference Governance schema.
    Raises pydantic.ValidationError explicitly to the API 400 Handler level.
    """
    governed_payload = CustomerInferencePayload(**json_dict)
    return governed_payload

