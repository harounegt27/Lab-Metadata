import re


def extract_invoice_metadata(text: str) -> dict:
    # Normalize all whitespace into single spaces
    flat = re.sub(r"\s+", " ", text).strip()

    def find(pattern):
        match = re.search(pattern, flat, re.IGNORECASE)
        return match.group(1).strip() if match else None

    # ── values that appear BEFORE their label ──────────────────────────────
    # "$634.89 Date : Ship Mode : Balance Due :"
    # → grab the value that sits just before "Date :"
    total_before_date = find(r"\$([\d,]+\.\d{2})\s+Date\s*:")

    # "$551.16 $83.73 $634.89 Subtotal : Shipping : Total :"
    # layout: subtotal  shipping  total  Subtotal: Shipping: Total:
    # grab the 3 consecutive dollar amounts before "Subtotal :"
    three_amounts = re.search(
        r"\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})\s+Subtotal",
        flat, re.IGNORECASE
    )
    # with discount: 4 amounts before Subtotal
    four_amounts = re.search(
        r"\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})\s+Subtotal",
        flat, re.IGNORECASE
    )

    if four_amounts:
        # subtotal  discount  shipping  total
        amount_val   = four_amounts.group(1)
        discount_val = four_amounts.group(2)
        shipping_val = four_amounts.group(3)
        total_val    = four_amounts.group(4)
    elif three_amounts:
        # subtotal  shipping  total  (no discount)
        amount_val   = three_amounts.group(1)
        discount_val = None
        shipping_val = three_amounts.group(2)
        total_val    = three_amounts.group(3)
    else:
        amount_val = discount_val = shipping_val = total_val = None

    # fallback total from before "Date :"
    if not total_val:
        total_val = total_before_date

    return {
        # "# 20581"
        "invoice_number": find(r"#\s*(\d{1,6})"),

        # "Nov 08 2012" — comes right before "Date :"
        "date": find(
            r"([A-Za-z]{3}\s+\d{1,2}\s+\d{4})\s+(?:First|Second|Same|Standard)"
        ),

        # "Bill To : Alejandro Savely" → stop before "Ship To"
        "bill_to": find(r"Bill\s*To\s*:\s*(.+?)\s+Ship\s*To"),

        # "Ship To : Kanpur, Uttar Pradesh, India" → stop before month name
        "ship_to": find(
            r"Ship\s*To\s*:\s*(.+?)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        ),

        # Product name: after "Amount" and before quantity
        "item": find(r"Amount\s+(.+?)\s+\d+\s+\$[\d,]+\.\d{2}"),

        # Quantity
        "quantity": find(r"Amount\s+.+?\s+(\d+)\s+\$[\d,]+\.\d{2}"),

        # Financial fields — all parsed from position-based groups above
        "amount":   amount_val,
        "discount": discount_val,
        "shipping": shipping_val,
        "total":    total_val,
    }


if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFLoader
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"invoices\invoice_Alejandro Savely_20581.pdf"
    docs = PyPDFLoader(path).load()
    text = "\n".join(d.page_content for d in docs)

    flat = re.sub(r"\s+", " ", text).strip()
    print("── FLAT TEXT ──────────────────────────────────────")
    print(flat)
    print("\n── EXTRACTED METADATA ─────────────────────────────")
    meta = extract_invoice_metadata(text)
    for k, v in meta.items():
        status = "✓" if v else "✗ None"
        print(f"  {k:20} {status:8} → {v}")