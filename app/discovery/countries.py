"""Kraje, w ktorych szukamy stron z przepisami: jezyk wyszukiwania + zapasowe zapytania (gdy brak LLM)."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Country:
    code: str            # ISO 3166-1 alpha-2 (maly), np. "ua"
    name: str            # nazwa po polsku (panel)
    language: str        # jezyk wynikow w wyszukiwarce, np. "uk-UA"
    language_name: str   # nazwa jezyka dla LLM, np. "Ukrainian"
    fallback_queries: tuple[str, ...] = field(default_factory=tuple)


COUNTRIES: dict[str, Country] = {c.code: c for c in [
    Country("ua", "Ukraina", "uk-UA", "Ukrainian",
            ("народні рецепти лікарські рослини", "настоянка з трав рецепт", "мазь з трав своїми руками", "фарбування тканин рослинами", "натуральна фарба для волосся трави")),
    Country("de", "Niemcy", "de-DE", "German",
            ("Heilkräuter Rezept Tinktur", "Kräutersalbe selber machen", "Pflanzenfarben Wolle färben", "Haare natürlich färben Kräuter", "Kräuterlikör selber machen")),
    Country("at", "Austria", "de-AT", "German",
            ("Hausmittel Kräuter Rezept", "Kräutersalbe selber machen", "Pflanzenfarben färben")),
    Country("fr", "Francja", "fr-FR", "French",
            ("remède de grand-mère plantes", "teinture mère maison recette", "teinture végétale laine", "coloration cheveux naturelle plantes", "baume aux plantes recette")),
    Country("es", "Hiszpania", "es-ES", "Spanish",
            ("remedios caseros con plantas medicinales", "tintura madre casera receta", "tintes naturales con plantas", "ungüento de hierbas casero")),
    Country("it", "Wlochy", "it-IT", "Italian",
            ("rimedi naturali erbe ricetta", "tintura madre fatta in casa", "tintura naturale lana piante", "unguento alle erbe fatto in casa", "liquore alle erbe fatto in casa")),
    Country("cz", "Czechy", "cs-CZ", "Czech",
            ("bylinná tinktura recept", "bylinná mast domácí", "barvení vlny rostlinami", "babské rady byliny")),
    Country("sk", "Slowacja", "sk-SK", "Slovak",
            ("bylinná tinktúra recept", "bylinná masť domáca")),
    Country("lt", "Litwa", "lt-LT", "Lithuanian",
            ("vaistažolių tinktūra receptas", "liaudies medicina žolelės")),
    Country("hu", "Wegry", "hu-HU", "Hungarian",
            ("gyógynövény tinktúra recept", "házi gyógynövényes krém")),
    Country("ro", "Rumunia", "ro-RO", "Romanian",
            ("tinctură din plante rețetă", "leacuri băbești plante")),
    Country("bg", "Bulgaria", "bg-BG", "Bulgarian",
            ("билкова тинктура рецепта", "народна медицина билки рецепти")),
    Country("hr", "Chorwacja", "hr-HR", "Croatian",
            ("ljekovito bilje tinktura recept", "domaća mast od bilja")),
    Country("rs", "Serbia", "sr-RS", "Serbian",
            ("lekovito bilje tinktura recept", "narodni lekovi bilje")),
    Country("gb", "Wielka Brytania", "en-GB", "English",
            ("herbal remedy recipes", "herbal tincture recipe", "natural plant dyeing wool", "homemade herbal salve", "natural hair dye herbs")),
    Country("us", "USA", "en-US", "English",
            ("herbal remedy recipes", "herbal tincture recipe", "natural dye with plants fabric", "homemade herbal salve recipe", "herbal hair rinse recipe")),
    Country("pl", "Polska", "pl-PL", "Polish",
            ("nalewka z ziół przepis", "maść ziołowa domowa", "farbowanie tkanin roślinami", "naturalna farba do włosów zioła", "domowe leki z ziół")),
]}


def get_country(code: str | None) -> Country | None:
    return COUNTRIES.get((code or "").lower())
