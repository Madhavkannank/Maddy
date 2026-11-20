"""
Math & Technical Phonetics Engine for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You)
Translates raw mathematical expressions, SymPy symbolic solutions, and ASCII symbols
into elegant, natural spoken English while maintaining clean Unicode formatting for display.
"""
import os
import re
from typing import List, Optional, Set, Tuple


class MathPhoneticizer:
    """
    Normalizes and phoneticizes mathematical outputs for voice synthesis (TTS)
    and produces clean Unicode formatted strings for display/logs.
    """

    _DICT_WORDS: Optional[Set[str]] = None

    @classmethod
    def _get_dictionary_words(cls) -> set:
        """Lazily load system English dictionary words for dynamic segmentation."""
        if cls._DICT_WORDS is not None:
            return cls._DICT_WORDS

        words = set()
        dict_paths = [
            "/usr/share/dict/words",
            "/usr/dict/words",
            "/etc/dictionaries-common/words",
        ]
        for p in dict_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        words = {line.strip().lower() for line in f if len(line.strip()) >= 2}
                    if words:
                        break
                except Exception:
                    pass

        domain_vocabulary = {
            "maddy", "clash", "clasher", "dodi", "repack", "repacks", "fit", "girl",
            "fitgirl", "spotify", "search", "said", "open", "play", "track", "music",
            "song", "game", "steam", "code", "browser", "window", "terminal", "screen",
            "system", "audio", "voice", "video", "photo", "camera", "calc", "calculator"
        }
        words.update(domain_vocabulary)

        cls._DICT_WORDS = words
        return cls._DICT_WORDS

    @classmethod
    def _segment_compound_token(cls, token: str) -> list[str]:
        """Dynamically segments a glued token using dictionary morphology without hardcoded site lists."""
        clean = token.strip()
        lower = clean.lower()
        if len(lower) < 6:
            return [clean.capitalize() if len(clean) > 1 else clean]

        dict_words = cls._get_dictionary_words()
        if not dict_words:
            return [clean.capitalize()]

        # If the word itself is already a valid dictionary word, preserve it intact
        if lower in dict_words or (lower.endswith("s") and lower[:-1] in dict_words):
            return [clean.capitalize()]

        # Search for longest valid compound segmentation
        for i in range(3, len(lower) - 2):
            left = lower[:i]
            right = lower[i:]
            right_base = right[:-1] if right.endswith("s") else right
            if left in dict_words and (right in dict_words or right_base in dict_words):
                w1 = clean[:i].capitalize()
                w2 = clean[i:].capitalize()
                return [w1, w2]

        return [clean.capitalize() if len(clean) > 1 else clean]

    @staticmethod
    def normalize_spelled_letters(text: str) -> str:
        """
        Normalize spoken spelled-out letters into coherent words.
        Uses pause/spacing delimiters and dynamic morphological dictionary segmentation.
        e.g. 'M A D D Y   C L A S H E R' -> 'Maddy Clasher'
             'M A D D Y C L A S H E R' -> 'Maddy Clasher'
             'F I T G I R L' -> 'Fitgirl'
             'S P O T I F Y' -> 'Spotify'
             'D O D I   R E P A C K S' -> 'Dodi Repacks'
             'D O D I R E P A C K S' -> 'Dodi Repacks'
        """
        if not text:
            return ""

        def _collapse(match):
            raw = match.group(0)
            words = re.split(r'\s{2,}', raw.strip())
            collapsed_words = []
            for w in words:
                letters = re.findall(r'[A-Za-z0-9]', w)
                if letters:
                    word = ''.join(letters)
                    collapsed_words.append(word)

            final_words = []
            for cw in collapsed_words:
                segmented = MathPhoneticizer._segment_compound_token(cw)
                final_words.extend(segmented)

            return ' '.join(final_words)

        return re.sub(r'\b(?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9]\b', _collapse, text)

    @staticmethod
    def format_math_result(raw_result: str) -> Tuple[str, str]:
        """
        Takes raw math/SymPy result string (e.g. 'x = 1 - I*sqrt(2), x = 1 + I*sqrt(2)' or '1 +/- isqrt2')
        and returns a tuple of:
          (display_text, speech_text)
        """
        if not raw_result:
            return "", ""

        clean = raw_result.strip()

        # 1. Clean SymPy output into standard readable representation
        display = MathPhoneticizer.to_display_unicode(clean)

        # 2. Phoneticize for speech
        speech = MathPhoneticizer.to_spoken_english(clean)

        return display, speech

    @staticmethod
    def to_display_unicode(text: str) -> str:
        """Format raw ASCII math into clean Unicode for console/logs."""
        t = text
        # Handle SymPy complex roots in both orders: sqrt(2)*I and I*sqrt(2)
        t = re.sub(r"sqrt\((\d+)\)\*[iI]\b", r"i√\1", t)
        t = re.sub(r"\b[iI]\*sqrt\((\d+)\)", r"i√\1", t)
        t = re.sub(r"\bI\*sqrt", "i√", t)
        t = re.sub(r"\bisqrt\(?(\d+)\)?", r"i√\1", t)
        t = re.sub(r"sqrt\((\d+)\)", r"√\1", t)
        t = re.sub(r"√\((\d+)\)", r"√\1", t)
        t = re.sub(r"\bI\b", "i", t)

        # Normalize powers
        t = t.replace("^2", "²").replace("**2", "²")
        t = t.replace("^3", "³").replace("**3", "³")

        # Normalize plus/minus
        t = t.replace("+/-", "±")

        # Normalize multiplication
        t = re.sub(r"(\d)\s*\*\s*([a-zA-Z])", r"\1\2", t)
        t = t.replace("*", " × ")

        return t.strip()

    @staticmethod
    def to_spoken_english(text: str) -> str:
        """
        Converts mathematical expressions into fluid, natural English for TTS.
        e.g. 'x = 1 +/- isqrt2' -> 'x equals 1 plus or minus i times the square root of 2'
        """
        s = text.strip()

        # Handle complex imaginary square root combinations in both orders:
        # e.g. 'sqrt(2)*I', 'sqrt(2)*i', 'I*sqrt(2)', 'isqrt2', 'isqrt(2)'
        s = re.sub(r"\bsqrt\(?(\d+)\)?\s*\*\s*[iI]\b", r" i times the square root of \1 ", s)
        s = re.sub(r"\bsqrt\(?([a-zA-Z]+)\)?\s*\*\s*[iI]\b", r" i times the square root of \1 ", s)
        s = re.sub(r"\b(?:isqrt|I\*sqrt|i\*sqrt)\(?(\d+)\)?", r" i times the square root of \1 ", s)
        s = re.sub(r"\b(?:isqrt|I\*sqrt|i\*sqrt)\(?([a-zA-Z]+)\)?", r" i times the square root of \1 ", s)
        s = re.sub(r"i√(\d+)", r" i times the square root of \1 ", s)

        # Standard square roots: sqrt(144) -> 'the square root of 144'
        s = re.sub(r"\bsqrt\(?(\d+)\)?", r" the square root of \1 ", s)
        s = re.sub(r"\bsqrt\(?([a-zA-Z]+)\)?", r" the square root of \1 ", s)

        # Imaginary unit alone: '1 + I' -> '1 plus i'
        s = re.sub(r"\bI\b", "i", s)

        # Plus or minus
        s = re.sub(r"\s*(?:\+\/\-|±)\s*", " plus or minus ", s)

        # Powers and exponents
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^2|\*\*2|²)", r"\1 squared", s)
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^3|\*\*3|³)", r"\1 cubed", s)
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^|\*\*)(\d+)", r"\1 to the power of \2", s)

        # Square root symbol alone: √2 -> 'the square root of 2'
        s = re.sub(r"√(\d+)", r" the square root of \1 ", s)
        s = re.sub(r"√([a-zA-Z]+)", r" the square root of \1 ", s)

        # Relational and arithmetic operators (guarded against hyphenated words)
        s = re.sub(r"\s+=\s+", " equals ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])\s*=\s*(?=[a-zA-Z0-9\-])", " equals ", s)
        s = re.sub(r"\s+\+\s+", " plus ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])\+(?=[a-zA-Z0-9])", " plus ", s)
        s = re.sub(r"\s+\-\s+", " minus ", s)
        s = re.sub(r"(?<=\d)\-(?=\d)", " minus ", s)
        s = re.sub(r"\s+\*\s+", " times ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])\*(?=[a-zA-Z0-9])", " times ", s)
        # Mathematical division vs. title/conversational slashes
        # Year/Date ranges: e.g. 2023/2024 -> 2023 to 2024
        s = re.sub(r"\b(19\d\d|20\d\d)\/(19\d\d|20\d\d|\d{2})\b", r"\1 to \2", s)
        s = re.sub(r"\s*÷\s*", " divided by ", s)
        s = re.sub(r"(?<=\d)\s*\/\s*(?=\d)", " divided by ", s)
        # Slashes between words/titles (e.g. 'The Lord of the Rings / The Hobbit', 'Windows / Linux', 'and/or') -> ' or '
        s = re.sub(r"(?<=[a-zA-Z])\s*\/\s*(?=[a-zA-Z])", " or ", s)
        s = re.sub(r"\s+\/\s+", " or ", s)

        # Normalize Whisper transcription slips in math
        s = re.sub(r"\b([a-zA-Z])\s+square\b", r"\1 squared", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-zA-Z])\s+cube\b", r"\1 cubed", s, flags=re.IGNORECASE)

        # Clean multiple spaces and dangling punctuation
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def sanitize_for_speech(text: str) -> str:
        """
        Thoroughly sanitize and normalize any text (assistant response, LLM output, math steps)
        before being sent to the TTS voice synthesizer.
        
        1. Strips LaTeX math delimiters: $$...$$ and $...$ (e.g. '$x$' -> 'x', '$6x + 5$' -> '6x plus 5').
        2. Replaces common LaTeX math commands with natural spoken English (\\frac, \\times, \\pm, etc.).
        3. Converts currency like '$50' into '50 dollars' while stripping non-currency dollar signs.
        4. Strips Markdown formatting (**, *, _, `, ```, ###, [text](url)).
        5. Phoneticizes mathematical operators (+, -, *, /, =, ^) when used in equations.
        6. Strips emojis and technical brackets that trip up TTS engines.
        """
        if not text:
            return ""

        s = text

        # 1. Clean JSON / HTTP technical blobs if accidentally passed
        if ("{" in s and "}" in s) and any(k in s for k in ["status_code", "detail", "error_code", "RateLimit"]):
            return "My neural connection is temporarily experiencing high traffic, boss."

        # 2. Currency conversion:
        # e.g. "$50" or "$20.50" -> "50 dollars", "20.50 dollars"
        s = re.sub(r"\$(\d+(?:\.\d+)?)\b", r"\1 dollars", s)

        # 3. LaTeX blocks $$...$$ and inline $...$
        s = re.sub(r"\$\$([^\$]+)\$\$", r" \1 ", s)
        s = re.sub(r"\$([^\$]+)\$", r" \1 ", s)

        # Stray dollar signs adjacent to variables or symbols: "$x", "$ x", "x$"
        s = re.sub(r"\$\s*([a-zA-Z])", r"\1", s)
        s = re.sub(r"([a-zA-Z])\s*\$", r"\1", s)
        s = s.replace("$", "")

        # 4. LaTeX command replacements
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r" \1 divided by \2 ", s)
        s = re.sub(r"\\sqrt\{([^}]+)\}", r" the square root of \1 ", s)
        s = s.replace(r"\times", " times ")
        s = s.replace(r"\cdot", " times ")
        s = s.replace(r"\pm", " plus or minus ")
        s = s.replace(r"\approx", " approximately ")
        s = s.replace(r"\leq", " less than or equal to ").replace(r"\le", " less than or equal to ")
        s = s.replace(r"\geq", " greater than or equal to ").replace(r"\ge", " greater than or equal to ")
        s = s.replace(r"\neq", " not equal to ")

        # 5. Markdown syntax cleanup
        # Code blocks ```python ... ``` -> ...
        s = re.sub(r"```[\w]*\n?(.*?)```", r"\1", s, flags=re.DOTALL)
        # Inline code `code`
        s = re.sub(r"`([^`]+)`", r"\1", s)
        # Markdown links [text](url) -> text
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
        # Strip all raw URLs (never spell out http:// or https:// over speech synthesis)
        s = re.sub(r"https?://\S+", "", s)
        s = re.sub(r"www\.\S+", "", s)
        # Bold and italic markdown: **text**, *text*, __text__, _text_
        s = re.sub(r"\*\*([^\*]+)\*\*", r"\1", s)
        s = re.sub(r"\*([^\*]+)\*", r"\1", s)
        s = re.sub(r"__([^_]+)__", r"\1", s)
        # Headings: ### Title -> Title
        s = re.sub(r"^\s*#{1,6}\s+", "", s, flags=re.MULTILINE)
        # Bullet points: * or - at line starts or following newlines
        s = re.sub(r"(?:^|\n)\s*[\*\-]\s+", "\n", s)
        # Markdown tables and pipe symbols: replace | with space so TTS never pronounces "pipe"
        s = s.replace("|", " ")

        # 6. Math symbol speech translation in text
        # Equals sign surrounded by numbers/letters
        s = re.sub(r"\s+=\s+", " equals ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])\s*=\s*(?=[a-zA-Z0-9\-])", " equals ", s)
        # Plus
        s = re.sub(r"\s+\+\s+", " plus ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])\+(?=[a-zA-Z0-9])", " plus ", s)
        # Minus (strictly arithmetic numbers or algebraic expressions, NOT dashes between words)
        s = re.sub(r"(?<=\d)\s*[\-−]\s*(?=\d)", " minus ", s)
        s = re.sub(r"(?<=[a-zA-Z])\s*[\-−]\s*(?=\d)", " minus ", s)
        s = re.sub(r"(?<=\d)\s*[\-−]\s*(?=[a-zA-Z])", " minus ", s)
        # Conversational dashes between words (e.g. 'Game - Edition', 'Title – Subtitle') -> natural pause
        s = re.sub(r"(?<=[a-zA-Z])\s*[-–—]\s*(?=[a-zA-Z])", ", ", s)
        # Times
        s = re.sub(r"\s+[\*×]\s+", " times ", s)
        s = re.sub(r"(?<=[a-zA-Z0-9])[\*×](?=[a-zA-Z0-9])", " times ", s)
        # Division: strictly arithmetic numbers or explicit division symbol
        # Year/Date ranges: e.g. 2023/2024 -> 2023 to 2024
        s = re.sub(r"\b(19\d\d|20\d\d)\/(19\d\d|20\d\d|\d{2})\b", r"\1 to \2", s)
        # Symbol ÷ is always division
        s = re.sub(r"\s*÷\s*", " divided by ", s)
        # Arithmetic division strictly between numbers
        s = re.sub(r"(?<=\d)\s*[\/]\s*(?=\d)", " divided by ", s)
        # Slashes between words/titles (e.g. 'The Lord of the Rings / The Hobbit', 'Windows / Linux', 'and/or') -> ' or '
        s = re.sub(r"(?<=[a-zA-Z])\s*\/\s*(?=[a-zA-Z])", " or ", s)
        # Slashes surrounded by spaces in text -> ' or '
        s = re.sub(r"\s+\/\s+", " or ", s)

        # Strip phantom 'and links' claim if no links were present in source text
        if not re.search(r"https?://|www\.", text, re.IGNORECASE):
            s = re.sub(r"\s+results\s+and\s+links\b", " results", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+details\s+and\s+links\b", " details", s, flags=re.IGNORECASE)
            s = re.sub(r"\band\s+links\b", "", s, flags=re.IGNORECASE)

        # Strip unnatural meta-commentary like 'listed below', 'links below', 'as listed below'
        s = re.sub(r"\b(?:as\s+)?(?:listed|provided|shown|given)\s+below\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:the\s+)?links\s+(?:are\s+)?(?:listed\s+)?below\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:links|details|results)\s+below\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:i\s+(?:have\s+)?(?:listed|provided|included)\s+(?:the\s+)?(?:links|details)\s+below[^\.\n]*[\.\n]?)", "", s, flags=re.IGNORECASE)

        # Powers: x^2 or x² -> x squared
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^2|\*\*2|²)", r"\1 squared", s)
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^3|\*\*3|³)", r"\1 cubed", s)
        s = re.sub(r"([a-zA-Z0-9]+)(?:\^|\*\*)(\d+)", r"\1 to the power of \2", s)

        # 6.5. Sequence & Counting Number Normalization for Neural TTS
        # In comma-separated digit lists (e.g. "1, 2, 3, 4, 5..."), replace commas with periods
        # so speech synthesizers take natural prosodic breaths and don't slur or run out of breath.
        def _replace_number_sequence_commas(match):
            return re.sub(r'(?<=\d),\s+(?=\d)', '. ', match.group(0))
        s = re.sub(r'\b\d+(?:,\s+\d+){2,}\b', _replace_number_sequence_commas, s)

        # 7. Clean punctuation and redundant whitespace
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def extract_speakable_text(text: str) -> str:
        """
        Extract an executive, charismatic, speakable voice response from a full visual response.
        Eliminates raw URLs, excessive numbered lists, markdown tables, and phonetic clutter
        so that TTS sounds like an intelligent personal assistant rather than reading a database dump.
        """
        if not text or not text.strip():
            return ""

        raw_text = text.strip()

        # Check whether actual web links were present before URL stripping
        has_links = bool(
            re.search(r"https?://|www\.", raw_text, re.IGNORECASE)
            or re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", raw_text)
        )

        clean = raw_text

        # Clean JSON / technical errors
        if ("{" in clean and "}" in clean) and any(k in clean for k in ["status_code", "detail", "error_code", "RateLimit"]):
            return "My neural connection is temporarily experiencing high traffic, boss."

        # Convert markdown links [Title](URL) -> Title before stripping raw URLs
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)

        # Strip all raw URLs (including angle brackets or parenthesis enclosed)
        clean = re.sub(r'<https?://[^>]+>', '', clean)
        clean = re.sub(r'\(https?://[^\)]+\)', '', clean)
        clean = re.sub(r'https?://\S+', '', clean)
        clean = re.sub(r'www\.\S+', '', clean)

        # Case 0: Markdown table detection
        if re.search(r'\|\s*[\:\-]{3,}\s*\|', clean):
            raw_lines = clean.splitlines()
            table_rows = []
            prefix_lines = []
            suffix_lines = []
            found_table = False

            for line in raw_lines:
                line_s = line.strip()
                is_row = line_s.startswith('|') and line_s.endswith('|')
                is_sep = is_row and bool(re.match(r'^\|[\:\-\s\|]+\|$', line_s))

                if is_row:
                    found_table = True
                    if not is_sep:
                        cols = [c.strip() for c in line_s.strip('|').split('|')]
                        table_rows.append(cols)
                elif not found_table:
                    if line_s:
                        prefix_lines.append(line_s)
                else:
                    if line_s:
                        suffix_lines.append(line_s)

            items = []
            if table_rows:
                headers = [c.lower() for c in table_rows[0]]
                target_idx = 1 if len(table_rows[0]) > 1 else 0
                for kw in ['title', 'name', 'item', 'game', 'release', 'description', 'query', 'highlight']:
                    matched = False
                    for i, h in enumerate(headers):
                        if kw in h:
                            target_idx = i
                            matched = True
                            break
                    if matched:
                        break

                for row in table_rows[1:]:
                    if target_idx < len(row):
                        val = row[target_idx]
                    else:
                        val = max(row, key=len) if row else ""
                    val = re.sub(r'^\d+\s*[\-\–\—\:]\s*', '', val).strip()
                    clean_item = re.split(r'[\–\—]|(?<=\s)\-(?=\s)|\[|\(', val)[0].strip()
                    clean_item = MathPhoneticizer.sanitize_for_speech(clean_item)
                    if clean_item and len(clean_item) > 2 and clean_item.lower() not in ["n/a", "none", "status", "true", "false"]:
                        items.append(clean_item)

            prefix = " ".join(prefix_lines)
            suffix = " ".join(suffix_lines)
            suffix = re.sub(r'\b(?:at|on|in|to|for|with|from|by|of)\s*$', '', suffix).strip()

            res = []
            if prefix:
                clean_p = MathPhoneticizer.sanitize_for_speech(prefix).rstrip(".:;,")
                if clean_p:
                    res.append(clean_p + ".")
            if items:
                preview = ", and ".join(items[:3])
                res.append(f"Top entries include {preview}.")
            if suffix:
                clean_suf = MathPhoneticizer.sanitize_for_speech(suffix).rstrip(".:;,")
                if len(clean_suf) > 5:
                    res.append(clean_suf + ".")
            return " ".join(res)

        lines = [l.strip() for l in clean.splitlines() if l.strip()]
        if not lines:
            return ""

        first_line = lines[0]

        # Case 1: Search results, recommendations, or itemized lookup
        list_markers = [
            "result", "found", "matches", "repack", "here are", "top", "recommend",
            "suggest", "movies", "series", "shows", "titles", "games", "books", "options",
        ]
        is_list = any(marker in first_line.lower() for marker in list_markers) or (
            len(lines) > 2 and any(re.match(r'^(?:\d+[\.\)]|\*|\-)', re.sub(r'[\*_`]', '', l).strip()) for l in lines[1:])
        )

        if is_list:
            items = []
            for line in lines[1:]:
                clean_l = re.sub(r'[\*_`]', '', line).strip()
                m = re.match(r'^(?:\d+[\.\)]|\*|\-)\s*(.+)', clean_l)
                if m:
                    candidate = m.group(1).strip()
                    # Strip leading release ID / numbering (e.g. "1118- Call of Duty..." -> "Call of Duty...")
                    clean_c = re.sub(r'^\d+\s*[\-\–\—\:]\s*', '', candidate).strip()
                    # Strip subtitles or trailing explanations after long dashes, spaced hyphens, brackets, or parens
                    title = re.split(r'[\–\—]|(?<=\s)\-(?=\s)|\[|\(', clean_c)[0].strip()
                    title = MathPhoneticizer.sanitize_for_speech(title)
                    if title and len(title) > 2 and not title.lower().startswith("url"):
                        items.append(title)

            clean_lead = MathPhoneticizer.sanitize_for_speech(first_line).rstrip(".:;,")
            clean_lead = re.sub(r"\b(?:as\s+follows|listed\s+below|below|as\s+listed\s+below)\b", "", clean_lead, flags=re.IGNORECASE).rstrip(".:;, ")

            if items:
                preview = ", and ".join(items[:3])
                return f"{clean_lead}. Top entries include {preview}."
            return f"{clean_lead}."

        # Case 2: Multi-step technical list or commands
        if len(lines) > 3 and any(re.match(r'^(?:\d+[\.\)]|\*|\-)', re.sub(r'[\*_`]', '', l).strip()) for l in lines[1:]):
            clean_lead = MathPhoneticizer.sanitize_for_speech(first_line).rstrip(".:;,")
            clean_lead = re.sub(r"\b(?:as\s+follows|listed\s+below|below|as\s+listed\s+below)\b", "", clean_lead, flags=re.IGNORECASE).rstrip(".:;, ")
            return f"{clean_lead}."

        # Default: sanitize the whole text for speech
        return MathPhoneticizer.sanitize_for_speech(clean)

    @staticmethod
    def format_links_for_display(text: str) -> str:
        """
        Extract markdown links and raw URLs from response text and format them into
        a structured visual banner for clear, separated console display.
        """
        if not text:
            return ""

        found = []
        seen = set()

        # 1. Markdown formatted links: [Title](URL)
        for m in re.finditer(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', text):
            title = m.group(1).strip()
            url = m.group(2).strip()
            if url not in seen:
                seen.add(url)
                found.append((title, url))

        # 2. Raw standalone URLs: https://...
        for m in re.finditer(r'(?<!\()(https?://[^\s\)\'\"\]]+)', text):
            url = m.group(1).strip().rstrip('.,;:)\'\"<>')
            if url not in seen:
                seen.add(url)
                found.append(("", url))

        if not found:
            return ""

        lines = ["\n┌── 🔗 FETCHED LINKS & DESTINATIONS " + "─" * 38]
        for idx, (title, url) in enumerate(found, start=1):
            if title:
                lines.append(f"│ {idx}. {title}")
                lines.append(f"│    {url}")
            else:
                lines.append(f"│ {idx}. {url}")
        lines.append("└" + "─" * 70 + "\n")
        return "\n".join(lines)

