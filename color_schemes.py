"""
AHI Color Scheme Variants
Test different palettes for SBIR presentation by uncommenting the desired scheme.
"""

# =============================================================================
# SCHEME 1: CURRENT — Sage Green (Professional, Operational)
# =============================================================================
COLORS_SAGE_GREEN = {
    'app_bg': '#24282D',
    'card_bg': '#161b22',
    'sidebar_bg': '#0d1117',
    'elevated_bg': '#1c2128',
    'primary': '#4a7c59',        # Sage green
    'primary_light': '#6b9e7a',
    'primary_dark': '#2d5a3a',
    'accent': '#8fbc8f',         # Dark sea green
    'border': '#30363d',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    # Hazard-specific
    'fire': '#e05252',
    'flood': '#4a90d9',
    'wind': '#9b59b6',
    'winter': '#2ec4b6',
    'seismic': '#e67e22',
}

# =============================================================================
# SCHEME 2: NAVY-TEAL — Technical, Scientific (AI/ML forward for SBIR)
# =============================================================================
COLORS_NAVY_TEAL = {
    'app_bg': '#0f1419',         # Darker navy
    'card_bg': '#1a1f2e',        # Navy card
    'sidebar_bg': '#0d1117',
    'elevated_bg': '#252d3f',    # Slightly elevated navy
    'primary': '#0ea5e9',        # Bright teal/cyan
    'primary_light': '#38bdf8',  # Light teal
    'primary_dark': '#0369a1',   # Dark navy-blue
    'accent': '#06b6d4',         # Cyan accent
    'border': '#334155',         # Slate gray
    'text_primary': '#f1f5f9',   # Bright white
    'text_secondary': '#cbd5e1',
    'text_tertiary': '#94a3b8',
    # Hazard-specific (match energy)
    'fire': '#ef4444',           # Red
    'flood': '#3b82f6',          # Bright blue
    'wind': '#a855f7',           # Purple
    'winter': '#14b8a6',         # Teal
    'seismic': '#f59e0b',        # Amber
}

# =============================================================================
# SCHEME 3: NAVY-GRAY — Enterprise, Stable (Corporate credibility for SBIR)
# =============================================================================
COLORS_NAVY_GRAY = {
    'app_bg': '#111827',         # Very dark navy
    'card_bg': '#1f2937',        # Dark gray-blue
    'sidebar_bg': '#0d1117',
    'elevated_bg': '#374151',    # Elevated gray
    'primary': '#3b82f6',        # Professional blue
    'primary_light': '#60a5fa',  # Light blue
    'primary_dark': '#1e40af',   # Dark blue
    'accent': '#8b5cf6',         # Purple accent
    'border': '#4b5563',         # Muted gray
    'text_primary': '#f3f4f6',   # Off-white
    'text_secondary': '#d1d5db',
    'text_tertiary': '#9ca3af',
    # Hazard-specific
    'fire': '#dc2626',
    'flood': '#2563eb',
    'wind': '#7c3aed',
    'winter': '#0891b2',
    'seismic': '#d97706',
}


# =============================================================================
# INSTRUCTIONS FOR TESTING
# =============================================================================
"""
To test a color scheme:

1. In app.py, replace the COLORS dictionary initialization:
   
   Current:
   >>> COLORS = {
   >>>     'app_bg': '#24282D',
   ...
   
   Test Sage Green (current):
   >>> from color_schemes import COLORS_SAGE_GREEN
   >>> COLORS = COLORS_SAGE_GREEN
   
   Test Navy-Teal:
   >>> from color_schemes import COLORS_NAVY_TEAL
   >>> COLORS = COLORS_NAVY_TEAL
   
   Test Navy-Gray:
   >>> from color_schemes import COLORS_NAVY_GRAY
   >>> COLORS = COLORS_NAVY_GRAY

2. Run `streamlit run app.py` and review the visual feel
3. Compare which best conveys:
   - Scientific authority
   - Modern innovation
   - Government/operational trust
   - Commercial viability

SBIR REVIEWER SIGNALS:
- Navy-Teal: "This is cutting-edge AI tech" (Venture/Innovation focus)
- Navy-Gray: "This is enterprise-grade and stable" (Government/Adoption focus)
- Sage-Green: "This is operational/field-tested" (Emergency Mgmt authenticity)
"""
