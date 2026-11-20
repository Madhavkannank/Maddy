"""
E-Commerce Product Search and Filter Engine for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You)
Parses natural language product queries with price and rating constraints
(e.g. "a good pendrive under 2k", "best wireless mouse under 1000 on amazon")
and constructs direct filtered search URLs on Amazon, Flipkart, etc.
"""
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple


class ProductSearchEngine:
    """
    Parses and routes shopping/product queries into filtered e-commerce URLs.
    """

    KNOWN_ECOMMERCE_SITES = {
        "amazon": "Amazon",
        "amazon.in": "Amazon",
        "amazon.com": "Amazon",
        "flipkart": "Flipkart",
        "ebay": "eBay",
    }

    CATEGORY_TOP_BRANDS = {
        "laptop": ["HP", "Dell", "Lenovo", "ASUS", "Apple", "Acer"],
        "gaming laptop": ["ASUS", "HP", "Lenovo", "Dell", "Acer", "MSI"],
        "phone": ["Apple", "Samsung", "OnePlus", "Google", "Motorola", "iQOO"],
        "smartphone": ["Apple", "Samsung", "OnePlus", "Google", "Motorola", "iQOO"],
        "mobile": ["Apple", "Samsung", "OnePlus", "Google", "Motorola", "iQOO"],
        "shoes": ["Nike", "Adidas", "Puma", "Reebok", "Skechers", "Asics"],
        "shoe": ["Nike", "Adidas", "Puma", "Reebok", "Skechers", "Asics"],
        "running shoes": ["Nike", "Adidas", "Puma", "Reebok", "Skechers", "Asics"],
        "sneakers": ["Nike", "Adidas", "Puma", "Reebok", "Converse", "Vans"],
        "sneaker": ["Nike", "Adidas", "Puma", "Reebok", "Converse", "Vans"],
        "watch": ["Apple", "Samsung", "Titan", "Fossil", "Noise", "boAt", "Fastrack"],
        "smartwatch": ["Apple", "Samsung", "Titan", "Fossil", "Noise", "boAt"],
        "smart watch": ["Apple", "Samsung", "Titan", "Fossil", "Noise", "boAt"],
        "headphones": ["Sony", "boAt", "JBL", "Sennheiser", "Bose", "OnePlus"],
        "headphone": ["Sony", "boAt", "JBL", "Sennheiser", "Bose", "OnePlus"],
        "earphones": ["Sony", "boAt", "JBL", "Sennheiser", "OnePlus", "Realme"],
        "earphone": ["Sony", "boAt", "JBL", "Sennheiser", "OnePlus", "Realme"],
        "earbuds": ["Apple", "Samsung", "Sony", "boAt", "OnePlus", "JBL"],
        "keyboard": ["Logitech", "Razer", "Corsair", "Keychron", "Redragon", "HP", "Dell"],
        "mouse": ["Logitech", "Razer", "HP", "Dell", "Corsair"],
        "wireless mouse": ["Logitech", "HP", "Dell", "Lenovo", "Razer"],
        "mechanical keyboard": ["Keychron", "Logitech", "Razer", "Corsair", "Redragon"],
        "tv": ["Samsung", "LG", "Sony", "TCL", "Xiaomi", "OnePlus"],
        "television": ["Samsung", "LG", "Sony", "TCL", "Xiaomi", "OnePlus"],
        "pendrive": ["SanDisk", "Samsung", "HP", "Kingston", "Strontium"],
        "pen drive": ["SanDisk", "Samsung", "HP", "Kingston", "Strontium"],
        "ssd": ["Samsung", "Crucial", "Western Digital", "Kingston", "SanDisk"],
    }

    KNOWN_BRANDS = [
        "Apple", "Samsung", "HP", "Dell", "Lenovo", "ASUS", "Acer", "Sony",
        "Nike", "Adidas", "Puma", "Reebok", "Skechers", "Asics", "Under Armour",
        "boAt", "JBL", "Sennheiser", "Bose", "Logitech", "Razer", "Corsair",
        "SanDisk", "Kingston", "Crucial", "Western Digital", "OnePlus", "Google",
        "Motorola", "Xiaomi", "Realme", "iQOO", "Titan", "Fastrack", "Fossil",
        "LG", "TCL", "MSI", "Keychron", "Redragon"
    ]

    def parse_price(self, price_str: str, multiplier_str: Optional[str] = None) -> Optional[int]:
        """Convert '2k', '1.5k', '500', '3,000', '50k', '1 lakh' into integer currency value."""
        if not price_str:
            return None
        try:
            clean_str = price_str.replace(",", "").strip()
            val = float(clean_str)
            mult = (multiplier_str or "").lower().strip()
            if mult in ("k", "thousand"):
                val *= 1000
            elif mult in ("l", "lakh", "lac", "lakhs"):
                val *= 100000
            return int(round(val))
        except (ValueError, TypeError):
            return None

    def parse_sort_preference(self, text: str) -> Optional[str]:
        """Extract user sorting preference (price_asc, price_desc, rating, newest, popularity)."""
        lower = text.lower()
        if re.search(r"\b(?:low(?:est)?\s+to\s+high|price\s+(?:low\s+to\s+high|ascending|asc)|lowest\s+price|cheapest(?:\s+first)?)\b", lower):
            return "price_asc"
        if re.search(r"\b(?:high(?:est)?\s+to\s+low|price\s+(?:high\s+to\s+low|descending|desc)|highest\s+price|expensive(?:\s+first)?)\b", lower):
            return "price_desc"
        if re.search(r"\b(?:rating|ratings|review|reviews|customer\s+reviews?)\b", lower) and re.search(r"\b(?:sort|order|rank|by)\b", lower):
            return "rating"
        if re.search(r"\b(?:top\s+rated|highest\s+rating|best\s+reviews?)\b", lower) and "sort" in lower:
            return "rating"
        if re.search(r"\b(?:newest|latest|new\s+arrivals?|recency)\b", lower):
            return "newest"
        if re.search(r"\b(?:popularity|popular|best\s+sellers?|most\s+bought|featured)\b", lower):
            return "popularity"
        return None

    def clean_product_text(self, text: str) -> str:
        """
        Thoroughly purge price, rating, sorting, brand descriptors, site names,
        and conversational fluff from search queries, leaving only the pure product keywords.
        """
        working = text
        # Remove leading action phrases
        working = re.sub(
            r"^(?:find\s+(?:me\s+)?(?:a\s+product\s+called\s+|a\s+product\s+like\s+|a\s+|an\s+)?|search\s+(?:for\s+)?(?:a\s+|an\s+)?|look\s+up\s+(?:a\s+|an\s+)?|buy\s+(?:me\s+)?(?:a\s+|an\s+)?|show\s+(?:me\s+)?(?:a\s+|an\s+)?)",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Remove site suffixes/prefixes
        working = re.sub(r"\b(?:on|in)\s+(?:amazon|flipkart|ebay|the\s+site|the\s+website)\b", "", working, flags=re.IGNORECASE).strip()
        working = re.sub(r"\b(?:amazon|flipkart|ebay)\b", "", working, flags=re.IGNORECASE).strip()

        # Remove delivery phrases
        working = re.sub(
            r"\b(?:and\s+)?(?:deliverable\s+in\s+\d+\s+days?|deliver(?:y)?\s+(?:in|within)\s+\d+\s+days?|deliver(?:y)?\s+(?:by\s+)?tomorrow|fast\s+deliver(?:y)?|prime|same\s+day\s+deliver(?:y)?|next\s+day\s+deliver(?:y)?)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Remove price phrases including currency words and multipliers
        working = re.sub(
            r"\b(?:under|below|less\s+than|budget(?:\s+of)?|max)\s+(?:rs\.?|inr|₹|\$)?\s*[0-9,]+(?:\.[0-9]+)?(?:\s*(?:k|thousand|l|lakh|lac|lakhs))?(?:\s*(?:rupees|rs\.?|inr|bucks|dollars))?\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\bbetween\s+(?:rs\.?|inr|₹|\$)?\s*[0-9,]+(?:\.[0-9]+)?(?:\s*(?:k|thousand|l|lakh|lac|lakhs))?\s+(?:and|to)\s+(?:rs\.?|inr|₹|\$)?\s*[0-9,]+(?:\.[0-9]+)?(?:\s*(?:k|thousand|l|lakh|lac|lakhs))?(?:\s*(?:rupees|rs\.?|inr|bucks|dollars))?\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\b[0-9,]+(?:\.[0-9]+)?\s*(?:k|thousand|l|lakh|lac|lakhs)\s*(?:rupees|rs\.?|inr|bucks|dollars)?\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\b(?:rs\.?|inr|₹|\$)\s*[0-9,]+\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Remove sorting phrases
        working = re.sub(
            r"\b(?:and\s+)?(?:sort(?:ed)?(?:\s+stuff)?\s+by|sort(?:ed)?(?:\s+stuff)?|order\s+by)\s+(?:price\s+)?(?:low\s+to\s+high|high\s+to\s+low|lowest|highest|rating|ratings|review|reviews|newest|latest|popularity|popular)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\b(?:and\s+)?(?:sort(?:ed)?(?:\s+stuff)?(?:\s+by)?|order\s+by|rank\s+by)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\b(?:low\s+to\s+high|high\s+to\s+low|cheapest\s+first|lowest\s+price|highest\s+price)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Remove rating adjectives
        working = re.sub(
            r"\b(?:with\s+)?(?:a\s+good|the\s+best|good|best|top\s+rated|high\s+rated|highly\s+rated|[3-5]\s*stars?)(?:\s+rating|\s+ratings)?\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(
            r"\b(?:with\s+)?(?:rating|ratings)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Remove brand descriptors ("branded", "top brands", etc.)
        working = re.sub(
            r"\b(?:branded|top\s+brands?|best\s+brands?|popular\s+brands?|good\s+brands?)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()
        working = re.sub(r"\b(?:brand|brands)\b", "", working, flags=re.IGNORECASE).strip()

        # Remove conversational filtering phrases
        working = re.sub(
            r"\b(?:and\s+)?(?:filter(?:\s+out)?(?:\s+only)?|filter\s+by|filter|show\s+only|only)\b",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

        # Clean leftover conversational connectors
        working = re.sub(r"\bwith\b", " ", working, flags=re.IGNORECASE)
        working = re.sub(r"\band\b", " ", working, flags=re.IGNORECASE)
        working = re.sub(r"[,]+", " ", working)
        working = re.sub(r"^(?:a|an|the|some)\s+", "", working, flags=re.IGNORECASE).strip()
        working = re.sub(r"\s+", " ", working).strip()

        # Deduplicate repeated words
        words = working.split()
        seen = set()
        dedup_words = []
        for w in words:
            stem = w.lower().rstrip("s")
            if stem not in seen or len(stem) <= 2:
                seen.add(stem)
                dedup_words.append(w)
        return " ".join(dedup_words).strip()

    def is_product_query(self, text: str) -> bool:
        """Determine whether the request is a shopping/product search with filters or intent."""
        lower = text.lower().strip()
        shopping_markers = [
            r"\b(?:under|below|less\s+than|budget(?:\s+of)?|max)\s+(?:rs\.?|inr|₹|\$)?\s*[0-9,]+",
            r"\bbetween\s+[0-9,]+.*?\s+and\s+[0-9,]+",
            r"\b(?:buy|order|purchase)\s+(?:a|an|the)?\s+",
            r"\b(?:find|search(?:\s+for)?)\s+(?:a\s+good|the\s+best|a|an)?\s+[a-zA-Z0-9_\- ]+?\s+(?:under|below|on\s+amazon|on\s+flipkart)",
            r"\b(?:amazon|flipkart|ebay)\b.*\b(?:product|item|price|rating|deal|buy)\b",
            r"\b(?:a\s+good|best|top\s+rated)\s+[a-zA-Z0-9_\- ]+?\s+(?:under|below|less\s+than)",
            r"\b(?:phone|smartphone|laptop|tablet|pc)\s+with\s+.*\b(?:ram|storage|gb|tb)\b",
            r"\b(?:deliverable\s+in|deliver(?:y)?\s+(?:in|within))\s+\d+\s+days?\b",
            r"\b(?:filter(?:\s+out)?|show\s+only|only)\s+.*\b(?:brand|brands|sandisk|samsung|apple|sony|hp|dell|lenovo|boat|logitech|asus|acer)\b",
            r"\b(?:pen\s*drive|pendrive|flash\s*drive|usb\s*drive|hard\s*drive|ssd|laptop|phone|smartphone|mouse|keyboard|monitor|headphones|earphones|smartwatch|shoes|shoe|sneakers)\b.*\b(?:filter|only|brand|brands|branded|sandisk|samsung|apple|sony|hp|dell|lenovo|nike|adidas|puma)\b",
            r"\b(?:filter|only)\s+(?:for\s+)?(?:sandisk|samsung|apple|sony|hp|dell|lenovo|boat|logitech|asus|acer|nike|adidas|puma)\b",
            r"\b(?:sandisk|samsung|apple|sony|hp|dell|lenovo|boat|logitech|nike|adidas|puma)\s+(?:pen\s*drive|pendrive|phone|laptop|mouse|keyboard|ssd|shoes|shoe)\b",
            r"\b(?:branded|brand|brands|top\s+brands?)\s+[a-zA-Z0-9_\- ]+",
            r"\b(?:sort\s+by|sort\s+stuff|low\s+to\s+high|high\s+to\s+low|cheapest\s+first)\b",
            r"\b(?:4\s*star|5\s*star|rating)\b.*\b(?:laptop|phone|shoes|watch|headphones|pendrive|keyboard|mouse|amazon|flipkart)\b",
        ]
        return any(re.search(pat, lower) for pat in shopping_markers)

    def parse_product_search(self, text: str, model_provider: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Parse shopping query into:
        - product: clean search term without query fluff (e.g. 'motorola phone 8gb ram 128gb')
        - max_price / min_price: integer constraints
        - min_rating: 4 if 'good' / 'best' / 'top rated' specified
        - sort_by: 'price_asc', 'price_desc', 'rating', 'newest', 'popularity'
        - is_branded: True if user wants top/branded products
        - brands: detected specific brands
        - delivery_days: integer constraint (e.g. 3) if specified
        - has_fast_delivery: True if fast delivery / prime / X days specified
        - site: 'Amazon', 'Flipkart', etc.
        - url: Direct search URL with price, rating, brand, and delivery filter parameters
        """
        lower = text.lower().strip().rstrip(".?!")
        lower = re.sub(r"\b(?:pin\s*guy|pen\s*guy|pin\s*dive|pin\s*drive)\b", "pendrive", lower, flags=re.IGNORECASE)

        # 1. Detect browser preference if specified
        browser_target = "chrome" if "chrome" in lower else ("firefox" if "firefox" in lower else ("edge" if "edge" in lower else "browser"))

        # Strip browser indicators
        cleaned = re.sub(
            r"\b(?:in|on|using|with)\s+(?:the\s+)?(?:google\s+chrome|chrome|firefox|edge|browser)\b",
            "",
            lower,
        ).strip()

        # 2. Detect E-commerce destination
        site_name = "Amazon"
        if "flipkart" in cleaned:
            site_name = "Flipkart"
        elif "ebay" in cleaned:
            site_name = "eBay"
        elif "amazon" in cleaned:
            site_name = "Amazon"

        # 3. Detect Rating Constraint
        has_rating_filter = bool(re.search(r"\b(good|best|top\s+rated|high\s+rated|highly\s+rated|[3-5]\s*stars?)(?:\s+rating|\s+ratings)?\b", cleaned))

        # 4. Detect Sorting Preference
        sort_by = self.parse_sort_preference(cleaned)

        # 5. Detect Branded / Brand Filter Intent
        is_branded = bool(re.search(r"\b(?:branded|top\s+brands?|best\s+brands?|popular\s+brands?|good\s+brands?)\b", cleaned))

        # 6. Detect Specific Brands Mentioned
        detected_brands = [b for b in self.KNOWN_BRANDS if re.search(r"\b" + re.escape(b.lower()) + r"\b", cleaned)]

        # 7. Detect Delivery Constraints
        has_fast_delivery = bool(
            re.search(
                r"\b(?:deliverable\s+in\s+\d+\s+days?|deliver(?:y)?\s+(?:in|within)\s+\d+\s+days?|deliver(?:y)?\s+(?:by\s+)?tomorrow|fast\s+deliver(?:y)?|prime|same\s+day\s+deliver(?:y)?|next\s+day\s+deliver(?:y)?)\b",
                cleaned,
            )
        )
        delivery_days = None
        m_days = re.search(r"\b(?:deliverable\s+in|deliver(?:y)?\s+(?:in|within))\s+(\d+)\s+days?\b", cleaned)
        if m_days:
            delivery_days = int(m_days.group(1))

        # 8. Detect Price Constraints
        max_price = None
        min_price = None

        m_between = re.search(
            r"\bbetween\s+(?:rs\.?|inr|₹|\$)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(k|thousand|l|lakh|lac|lakhs)?\s*(?:and|to)\s+(?:rs\.?|inr|₹|\$)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(k|thousand|l|lakh|lac|lakhs)?(?:\s*(?:rupees|rs\.?|inr|bucks|dollars))?\b",
            cleaned,
        )
        if m_between:
            min_price = self.parse_price(m_between.group(1), m_between.group(2))
            max_price = self.parse_price(m_between.group(3), m_between.group(4))
        else:
            m_under = re.search(
                r"\b(?:under|below|less\s+than|budget(?:\s+of)?|max)\s+(?:rs\.?|inr|₹|\$)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(k|thousand|l|lakh|lac|lakhs)?(?:\s*(?:rupees|rs\.?|inr|bucks|dollars))?\b",
                cleaned,
            )
            if m_under:
                max_price = self.parse_price(m_under.group(1), m_under.group(2))

        # 9. Extract and Clean Core Product Keyword
        product = self.clean_product_text(cleaned)

        if not product or product in ("product", "item", "something"):
            return None

        slots = {
            "product": product,
            "target_site": site_name,
            "max_price": max_price,
            "min_price": min_price,
            "has_rating_filter": has_rating_filter,
            "sort_by": sort_by,
            "is_branded": is_branded,
            "brands": detected_brands,
            "has_fast_delivery": has_fast_delivery,
            "delivery_days": delivery_days,
        }
        return self.build_search_from_slots(slots, text, browser_target)

    def build_search_from_slots(
        self,
        slots: Dict[str, Any],
        original_request: str = "",
        browser_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build precision-filtered shopping search dict directly from structured slots.
        Strictly strips query fluff (price, rating, sorting, brand qualifiers) from
        the search keyword so that only the pure item name is queried, while applying
        money, brand, rating, and sorting as native e-commerce URL facet parameters.
        """
        orig_lower = (original_request or "").lower()

        # 1. Determine site
        site_raw = str(slots.get("target_site") or slots.get("site") or "Amazon").lower()
        if "flipkart" in site_raw or "flipkart" in orig_lower:
            site_name = "Flipkart"
        elif "ebay" in site_raw or "ebay" in orig_lower:
            site_name = "eBay"
        else:
            site_name = "Amazon"

        # 2. Determine browser
        if not browser_target or browser_target == "browser":
            browser_target = "chrome" if "chrome" in orig_lower else ("firefox" if "firefox" in orig_lower else ("edge" if "edge" in orig_lower else "browser"))
        browser_disp = browser_target.capitalize()

        # 3. Detect sorting and branding preferences
        sort_by = slots.get("sort_by") or self.parse_sort_preference(orig_lower)
        is_branded = bool(slots.get("is_branded") or re.search(r"\b(?:branded|top\s+brands?|best\s+brands?|popular\s+brands?|good\s+brands?)\b", orig_lower))

        # 4. Extract and sanitize core product keywords
        raw_product = str(slots.get("product_category") or slots.get("product") or slots.get("search_query") or "").strip()
        if not raw_product:
            raw_product = orig_lower

        cleaned_product = self.clean_product_text(raw_product)

        # Handle brands
        brands = slots.get("brands") or []
        if isinstance(brands, str):
            brands = [b.strip() for b in brands.split(",") if b.strip()]

        # Handle specs
        specs = slots.get("specs") or []
        if isinstance(specs, str):
            specs = [s.strip() for s in specs.split(",") if s.strip()]

        working_parts = [cleaned_product] if cleaned_product else []
        prod_lower = cleaned_product.lower()

        # If caller explicitly provided brands in slots (e.g. specific test payloads), include them in keywords
        if slots.get("brands"):
            for b in brands:
                if b and b.lower() not in prod_lower:
                    working_parts.append(b)

        for s in specs:
            if s and s.lower() not in prod_lower:
                working_parts.append(s)

        combined_text = " ".join(working_parts).strip()
        combined_text = re.sub(r"\b(?:pin\s*guy|pen\s*guy|pin\s*dive|pin\s*drive)\b", "pendrive", combined_text, flags=re.IGNORECASE)
        product = self.clean_product_text(combined_text) or "products"

        # 5. Determine active brand facet filters
        brands_to_filter = list(brands)
        if not brands_to_filter and is_branded:
            prod_key = product.lower().strip()
            for cat, b_list in self.CATEGORY_TOP_BRANDS.items():
                if cat in prod_key or prod_key in cat:
                    brands_to_filter = list(b_list)
                    break
            if not brands_to_filter:
                for cat, b_list in self.CATEGORY_TOP_BRANDS.items():
                    if any(w in cat for w in prod_key.split()):
                        brands_to_filter = list(b_list)
                        break

        # 6. Extract price constraints
        min_price = slots.get("min_price")
        max_price = slots.get("max_price")
        if isinstance(min_price, str):
            min_price = self.parse_price(min_price)
        elif min_price is not None:
            try:
                min_price = int(round(float(min_price)))
            except (ValueError, TypeError):
                min_price = None

        if isinstance(max_price, str):
            max_price = self.parse_price(max_price)
        elif max_price is not None:
            try:
                max_price = int(round(float(max_price)))
            except (ValueError, TypeError):
                max_price = None

        # 7. Extract rating constraint
        min_rating = slots.get("min_rating")
        has_rating_filter = bool(min_rating or slots.get("has_rating_filter") or re.search(r"\b(good|best|top\s+rated|high\s+rated|highly\s+rated|[3-5]\s*stars?)(?:\s+rating|\s+ratings)?\b", orig_lower))

        # 8. Extract delivery constraints
        has_fast_delivery = bool(
            slots.get("fast_delivery")
            or slots.get("has_fast_delivery")
            or slots.get("delivery_days")
            or re.search(r"\b(?:deliverable\s+in\s+\d+\s+days?|deliver(?:y)?\s+(?:in|within)\s+\d+\s+days?|deliver(?:y)?\s+(?:by\s+)?tomorrow|fast\s+deliver(?:y)?|prime)\b", orig_lower)
        )
        delivery_days = slots.get("delivery_days")
        if delivery_days is None:
            m_days = re.search(r"\b(?:deliverable\s+in|deliver(?:y)?\s+(?:in|within))\s+(\d+)\s+days?\b", orig_lower)
            if m_days:
                try:
                    delivery_days = int(m_days.group(1))
                except (ValueError, TypeError):
                    delivery_days = None
        else:
            try:
                delivery_days = int(delivery_days)
            except (ValueError, TypeError):
                delivery_days = None

        # 9. Build Target URL with native filters
        encoded_product = urllib.parse.quote_plus(product)

        if site_name == "Amazon":
            url = f"https://www.amazon.in/s?k={encoded_product}"
            params = []
            rh_parts = []

            # Rating filter: 4 stars & up (Amazon India: 1318476031)
            if has_rating_filter:
                rh_parts.append("p_72:1318476031")

            # Brand filter (Amazon p_89 facet)
            if brands_to_filter:
                rh_parts.append(f"p_89:{'|'.join(brands_to_filter)}")

            # Fast Delivery / Prime filter
            if has_fast_delivery:
                rh_parts.append("p_90:6741118031")  # Get it in 2-3 days / Tomorrow
                rh_parts.append("p_85:20912642031")  # Prime eligible

            # Price filter: high-price / low-price & rh price in paise (1 INR = 100 paise)
            if min_price is not None:
                params.append(f"low-price={min_price}")
            if max_price is not None:
                params.append(f"high-price={max_price}")

            # Amazon rh price range (p_36:min_paise-max_paise)
            if min_price is not None or max_price is not None:
                min_p = f"{min_price * 100}" if min_price is not None else ""
                max_p = f"{max_price * 100}" if max_price is not None else ""
                rh_parts.append(f"p_36:{min_p}-{max_p}")

            # Sorting filter
            if sort_by == "price_asc":
                params.append("s=price-asc-rank")
            elif sort_by == "price_desc":
                params.append("s=price-desc-rank")
            elif sort_by == "rating":
                params.append("s=review-rank")
            elif sort_by == "newest":
                params.append("s=date-desc-rank")
            elif sort_by == "popularity":
                params.append("s=popularity-rank")
            elif has_rating_filter:
                params.append("s=review-rank")

            if rh_parts:
                params.append(f"rh={urllib.parse.quote(','.join(rh_parts))}")

            if params:
                url += "&" + "&".join(params)

        elif site_name == "Flipkart":
            url = f"https://www.flipkart.com/search?q={encoded_product}"
            params = []
            if max_price is not None:
                params.append(f"p%5B%5D=facets.price_range.to%3D{max_price}")
            if min_price is not None:
                params.append(f"p%5B%5D=facets.price_range.from%3D{min_price}")
            if has_rating_filter:
                params.append("p%5B%5D=facets.rating%5B%5D%3D4%E2%98%85%20%26%20above")
            if has_fast_delivery:
                params.append("p%5B%5D=facets.serviceability%5B%5D%3Dexpress")
            if brands_to_filter:
                for b in brands_to_filter[:5]:
                    params.append(f"p%5B%5D=facets.brand%5B%5D%3D{urllib.parse.quote(b)}")
            if sort_by == "price_asc":
                params.append("sort=price_asc")
            elif sort_by == "price_desc":
                params.append("sort=price_desc")
            elif sort_by == "rating":
                params.append("sort=rating")
            elif sort_by == "newest":
                params.append("sort=recency_desc")
            elif sort_by == "popularity":
                params.append("sort=popularity")
            if params:
                url += "&" + "&".join(params)

        else:  # eBay
            url = f"https://www.ebay.com/sch/i.html?_nkw={encoded_product}"
            if max_price is not None:
                url += f"&_udhi={max_price}"
            if min_price is not None:
                url += f"&_udlo={min_price}"

        # 10. Formulate articulate confirmation messages
        filter_descriptions = []
        if brands_to_filter:
            if is_branded and not slots.get("brands"):
                filter_descriptions.append(f"top brands ({', '.join(brands_to_filter[:4])})")
            else:
                filter_descriptions.append(f"brand{'s' if len(brands_to_filter) > 1 else ''} {', '.join(brands_to_filter)}")

        if max_price is not None and min_price is not None:
            filter_descriptions.append(f"price between Rs. {min_price:,} and Rs. {max_price:,}")
        elif max_price is not None:
            filter_descriptions.append(f"price under Rs. {max_price:,}")
        elif min_price is not None:
            filter_descriptions.append(f"price above Rs. {min_price:,}")

        if has_rating_filter:
            filter_descriptions.append("4+ star ratings")

        if sort_by:
            sort_labels = {
                "price_asc": "sorted by price low-to-high",
                "price_desc": "sorted by price high-to-low",
                "rating": "sorted by customer reviews",
                "newest": "sorted by newest arrivals",
                "popularity": "sorted by popularity",
            }
            filter_descriptions.append(sort_labels.get(sort_by, f"sorted by {sort_by}"))

        if has_fast_delivery:
            if delivery_days:
                filter_descriptions.append(f"deliverable within {delivery_days} days")
            else:
                filter_descriptions.append("fast/prime delivery")

        filter_summary = " with " + " and ".join(filter_descriptions) + " applied" if filter_descriptions else ""

        action_desc = f"Opened {site_name} searching for '{product}'{filter_summary} in {browser_disp}."
        fallback = f"Searching for '{product}' on {site_name}{filter_summary}."

        return {
            "is_product_search": True,
            "product": product,
            "site": site_name,
            "browser": browser_target,
            "browser_disp": browser_disp,
            "max_price": max_price,
            "min_price": min_price,
            "has_rating_filter": has_rating_filter,
            "has_fast_delivery": has_fast_delivery,
            "delivery_days": delivery_days,
            "sort_by": sort_by,
            "is_branded": is_branded,
            "brands": brands_to_filter,
            "specs": specs,
            "url": url,
            "action_desc": action_desc,
            "fallback": fallback,
        }


    def advise_and_synthesize(
        self,
        user_request: str,
        parsed_product: Dict[str, Any],
        model_provider: Any,
        address: str = "boss",
    ) -> str:
        """
        Dynamically synthesize matching models via LLM for rich hardware/spec requests
        (e.g. Motorola phone with 8GB RAM, 128GB ROM under 20k, 3-day delivery).
        """
        product = parsed_product["product"]
        site = parsed_product["site"]
        max_p = parsed_product.get("max_price")
        deliv = parsed_product.get("delivery_days")

        has_specs = any(w in product.lower() for w in ["ram", "storage", "gb", "tb", "phone", "smartphone", "laptop", "pc", "monitor", "cpu", "gpu", "camera", "display", "processor", "spec", "specs"])
        is_conf = model_provider.is_configured() if hasattr(model_provider, "is_configured") else True
        if not has_specs or not model_provider or not is_conf:
            return parsed_product["fallback"]

        filter_txt = f" under ₹{max_p:,}" if max_p else ""
        default_resp = f"🛒 I have opened {site} on your PC searching for '{product}'{filter_txt}, {address}."

        prompt = (
            f"The user wants: \"{user_request}\"\n"
            f"Parsed product query: \"{product}\"\n"
            f"Max budget: ₹{max_p:,} if given\n" if max_p else ""
            f"Delivery: within {deliv} days / Prime\n" if deliv else ""
            f"Platform opened: {site}\n\n"
            f"IMPORTANT: You do NOT need to browse or fetch live prices yourself; M.A.D.D.Y. has ALREADY launched the live {site} website on the user's PC with their search query and budget filters applied. "
            f"As M.A.D.D.Y., respond to '{address}' in 1 or 2 concise, confident sentences. "
            f"State that you've opened {site} on their PC for '{product}'{filter_txt}, and briefly mention 1 or 2 notable models or series to look out for. "
            f"DO NOT include disclaimers about not having live web access or being an AI."
        )

        try:
            res = model_provider.generate(prompt=prompt)
            content = (res.content or "").strip()
            # Strict rejection of AI disclaimers or inability excuses
            disclaimers = [
                "cannot browse", "can't browse", "don't have access", "cannot access",
                "real-time", "as an ai", "as a language model", "sorry, i", "i cannot fetch"
            ]
            if content and not any(d in content.lower() for d in disclaimers) and not content.startswith("Neural brain service"):
                return content
        except Exception:
            pass

        return default_resp

    def synthesize_recommendations(
        self,
        product: str,
        max_price: Optional[int] = None,
        site: str = "Amazon",
        model_provider: Optional[Any] = None,
    ) -> List[Dict[str, str]]:
        """
        Dynamically synthesize top 3-4 recommended product options using the LLM.
        Does NOT rely on static hardcoded lists.
        """
        site_name = site if site in ("Amazon", "Flipkart") else "Amazon"

        def _make_link(query: str) -> str:
            eq = urllib.parse.quote_plus(query)
            if site_name == "Amazon":
                return f"https://www.amazon.in/s?k={eq}"
            return f"https://www.flipkart.com/search?q={eq}"

        # 1. Ask ModelProvider to synthesize real recommendations dynamically
        if model_provider and hasattr(model_provider, "is_configured") and model_provider.is_configured():
            budget_str = f"under ₹{max_price:,}" if max_price else "at best market price"
            prompt = (
                f"You are M.A.D.D.Y. product advisor. Recommend 3 to 4 specific top-rated real models or products for: '{product}' {budget_str} on {site_name}.\n"
                f"Respond ONLY with a valid JSON array of objects. Do not write markdown fences or any explanation.\n"
                f'Format: [{{"name": "Exact Brand and Model Name", "price": "~₹Price", "highlights": "Key feature or reason to buy"}}]\n'
            )
            try:
                res = model_provider.generate(prompt=prompt)
                raw = (res.content or "").strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()
                import json
                items = json.loads(raw)
                if isinstance(items, list) and items:
                    results = []
                    for it in items[:4]:
                        name = it.get("name", "").strip()
                        if name:
                            results.append({
                                "name": name,
                                "price": str(it.get("price", "")).strip(),
                                "highlights": str(it.get("highlights", "")).strip(),
                                "link": _make_link(name),
                            })
                    if results:
                        return results
            except Exception:
                pass

        # 2. Dynamic fallback without hardcoding specific brands
        return [
            {
                "name": f"Top Rated {product.title()}",
                "price": f"Under ₹{max_price:,}" if max_price else "Best Value",
                "highlights": f"Highest customer-reviewed items for '{product}' on {site_name}.",
                "link": _make_link(f"top rated {product}"),
            },
            {
                "name": f"Best Selling {product.title()}",
                "price": f"Under ₹{max_price:,}" if max_price else "Popular Choice",
                "highlights": f"Trending best sellers in '{product}' on {site_name}.",
                "link": _make_link(f"best seller {product}"),
            },
        ]

    def create_notepad_recommendations(
        self,
        recommendations: List[Dict[str, str]],
        query_desc: str,
        site: str = "Amazon",
    ) -> str:
        """
        Formats product recommendations into a temporary text file
        and launches it in notepad.exe non-blockingly on the Windows desktop.
        Returns the absolute path to the generated text file.
        """
        lines = [
            "=" * 65,
            f"  M.A.D.D.Y. PRODUCT RECOMMENDATIONS — {query_desc.upper()}",
            f"  Platform: {site} | Generated for your inspection",
            "=" * 65,
            "",
        ]

        for i, item in enumerate(recommendations, 1):
            lines.append(f"{i}. {item.get('name', 'Product')}")
            if item.get("price"):
                lines.append(f"   Estimated Price: {item['price']}")
            if item.get("highlights"):
                lines.append(f"   Key Highlights : {item['highlights']}")
            if item.get("link"):
                lines.append(f"   Direct Link    : {item['link']}")
            lines.append("")

        lines.extend([
            "-" * 65,
            "Tip: Click or copy any link above into your browser to inspect.",
            f"Say 'Next' to M.A.D.D.Y. anytime to advance to the next page of results.",
            "=" * 65,
        ])

        content = "\n".join(lines)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "maddy_product_recommendations.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Launch in notepad.exe non-blockingly
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["notepad.exe", file_path])
        except Exception:
            pass

        return file_path

    def build_page_url(self, base_url: str, site: str, page_number: int) -> str:
        """
        Constructs next page URL for Amazon / Flipkart search pagination.
        Does NOT alter active filters.
        """
        page_num = max(1, int(page_number))
        site_lower = site.lower().strip()

        if "amazon" in site_lower:
            # Replace existing &page=X or append &page=X
            if re.search(r"[?&]page=\d+", base_url):
                return re.sub(r"([?&]page=)\d+", rf"\g<1>{page_num}", base_url)
            sep = "&" if "?" in base_url else "?"
            return f"{base_url}{sep}page={page_num}"

        elif "flipkart" in site_lower:
            # Replace existing &page=X or append &page=X
            if re.search(r"[?&]page=\d+", base_url):
                return re.sub(r"([?&]page=)\d+", rf"\g<1>{page_num}", base_url)
            sep = "&" if "?" in base_url else "?"
            return f"{base_url}{sep}page={page_num}"

        # Default query parameter append
        if re.search(r"[?&]page=\d+", base_url):
            return re.sub(r"([?&]page=)\d+", rf"\g<1>{page_num}", base_url)
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}page={page_num}"
