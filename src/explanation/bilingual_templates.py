"""Bilingual explanation templates (English & Tamil) for electricity bill anomalies."""

from typing import Dict, Any, List


EXPLANATION_TEMPLATES_EN = {
    "CONFIRMED_ARITHMETIC_ERROR": (
        "CONFIRMED ARITHMETIC DISCREPANCY: The billed amount of ₹{billed_amount:.2f} deviates from the "
        "official statutory tariff calculation of ₹{calculated_amount:.2f} by ₹{discrepancy:.2f}. "
        "Slab-by-slab recomputation using verified TNERC tariff rules indicates an overcharge/undercharge. "
        "This is a verifiable mathematical calculation mismatch."
    ),
    "SUSPECTED_EXTENDED_CYCLE": (
        "BILLING CYCLE DURATION IRREGULARITY: The billing period was recorded across {billing_days} days "
        "instead of the standard 60-day bi-monthly cycle (+{excess_days} days delay). "
        "Without proportional slab proration, an extended cycle artificially pushes normal daily consumption "
        "into higher, more expensive tariff slabs. Impact: Estimated additional slab penalty of ~₹{penalty_est:.2f}."
    ),
    "UNUSUAL_CONSUMPTION_SURGE": (
        "UNUSUAL CONSUMPTION DETECTED: Recorded consumption ({units:.1f} units) represents a {spike_pct:.1f}% surge "
        "over your 6-month historical average ({hist_avg:.1f} units). "
        "While this may reflect heightened appliance/cooling usage during peak summer, the magnitude warrants physical "
        "meter inspection to rule out recording faults, neutral leakage, or wiring issues."
    ),
    "METER_READING_MISMATCH": (
        "METER READING INCONSISTENCY: The billed units ({billed_units}) do not match the physical difference "
        "between current reading ({curr_reading}) and previous reading ({prev_reading}), which equals {calc_units} units. "
        "Difference: {diff_units} units."
    ),
    "SUSPECTED_METER_FAULT": (
        "SUSPECTED METER ANOMALY: A disproportionate sudden surge ({jump_factor:.1f}x historical baseline) was recorded "
        "without seasonal or connected-load justification. Note: This finding indicates statistical abnormality; "
        "meter failure or malfunction cannot be proven without an official physical meter test (Rule 57 CEA regulations)."
    ),
    "DEAD_METER_ZERO": (
        "ZERO CONSUMPTION ALERT: 0 units recorded for premises with historical active consumption ({hist_avg:.1f} units). "
        "Possible stuck mechanical register, blown meter fuse, or unoccupied premises."
    ),
    "NORMAL_VERIFIED": (
        "BILL VERIFIED NORMAL: Meter arithmetic and tariff calculations match statutory TNERC orders precisely. "
        "Consumption is within expected seasonal variance."
    ),
}

EXPLANATION_TEMPLATES_TA = {
    "CONFIRMED_ARITHMETIC_ERROR": (
        "கட்டணக் கணக்கீட்டுப் பிழை உறுதி செய்யப்பட்டுள்ளது: மின்வாரிய ரசீதில் குறிப்பிடப்பட்டுள்ள தொகை ₹{billed_amount:.2f}, "
        "அதிகாரப்பூர்வ TNERC கட்டண விதிகளின்படி கணக்கிடப்பட்ட தொகையான ₹{calculated_amount:.2f}-ஐ விட ₹{discrepancy:.2f} "
        "வேறுபடுகிறது. இது அலகுக் கணக்கீட்டில் ஏற்பட்ட நேரடி கணிதப் பிழையாகும்."
    ),
    "SUSPECTED_EXTENDED_CYCLE": (
        "கணக்கீட்டுக் கால நீட்டிப்பு முரண்பாடு: வழக்கமான 60 நாட்களுக்குப் பதிலாக {billing_days} நாட்களுக்கு "
        "(+{excess_days} நாட்கள் தாமதம்) மின் நுகர்வு கணக்கிடப்பட்டுள்ளது. "
        "நாட்களுக்கு ஏற்ப கட்டண அடுக்குகள் (Slabs) மாற்றியமைக்கப்படாததால், சாதாரண நுகர்வும் அதிக கட்டண அடுக்குக்குள் "
        "செல்ல வழிவகுத்துள்ளது. உத்தேச கூடுதல் சுமை: சுமார் ₹{penalty_est:.2f}."
    ),
    "UNUSUAL_CONSUMPTION_SURGE": (
        "வழக்கத்திற்கு மாறான உயர் மின் நுகர்வு: தற்போதைய நுகர்வு ({units:.1f} அலகுகள்), முந்தைய 6 மாத சராசரி "
        "நுகர்வை ({hist_avg:.1f} அலகுகள்) விட {spike_pct:.1f}% அதிகரித்துள்ளது. "
        "கோடைக்கால குளிர்சாதனப் பயன்பாட்டால் இது ஏற்படலாம்; இருப்பினும் ஏதேனும் வயரிங் கசிவு அல்லது மின்மானி "
        "பிழை உள்ளதா என்பதை உறுதி செய்ய நேரடி ஆய்வு பரிந்துரைக்கப்படுகிறது."
    ),
    "METER_READING_MISMATCH": (
        "மின்மானி அளவீட்டு முரண்பாடு: ரசீதில் உள்ள அலகுகள் ({billed_units}), முந்தைய அளவீடு ({prev_reading}) மற்றும் "
        "தற்போதைய அளவீட்டின் ({curr_reading}) உண்மையான வித்தியாசமான {calc_units} அலகுகளுடன் பொருந்தவில்லை. "
        "வித்தியாசம்: {diff_units} அலகுகள்."
    ),
    "SUSPECTED_METER_FAULT": (
        "மின்மானி கோளாறு சந்தேகம்: வழக்கமான நுகர்வை விட திடீரென {jump_factor:.1f} மடங்கு உயர்வு பதிவாகியுள்ளது. "
        "குறிப்பு: இது புள்ளிவிவர அடிப்படையிலான எச்சரிக்கை மட்டுமே; மின்சார வாரியத்தின் நேரடி பரிசோதனை "
        "இன்றி மின்மானி பழுதாகிவிட்டதாகக் கூற முடியாது."
    ),
    "DEAD_METER_ZERO": (
        "சுழி நுகர்வு எச்சரிக்கை: தொடர் பயன்பாட்டில் உள்ள இணைப்பிற்கு (சராசரி {hist_avg:.1f} அலகுகள்) 0 அலகுகள் "
        "பதிவாகியுள்ளது. மின்மானி இயங்காமல் நின்றுவிட்டதா அல்லது சுவிட்ச் கோளாறா என்பதைச் சரிபார்க்கவும்."
    ),
    "NORMAL_VERIFIED": (
        "ரசீது சரிபார்க்கப்பட்டது (சரியானது): மின்மானி அளவீடுகள் மற்றும் தமிழ்நாடு மின்சார ஒழுங்குமுறை ஆணையத்தின் "
        "(TNERC) கட்டண அடுக்கு கணக்கீடுகள் அனைத்தும் துல்லியமாகப் பொருந்துகின்றன."
    ),
}
