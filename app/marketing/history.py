"""Historia edycji szkicow: wersje calej rodziny posta (poprawki wg raportu tworza nowe posty z parent_id)
i porownanie dwoch tekstow slowo po slowie."""
import difflib
import re

from sqlmodel import Session, col, select

from app.marketing.review import text_hash
from app.models import FbPost, FbPostReview, FbPostVersion

KIND_LABELS = {"generated": "napisany", "edit": "edycja ręczna", "rewrite": "przeredagowany wg raportu",
               "improve": "od nowa wg uwag", "round": "runda poprawek (odrzucona)", "restore": "przywrócony",
               "initial": "stan sprzed historii"}
_TOKEN = re.compile(r"\s+|[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ'-]+|[^\w\s]", re.U)


def root_of(post: FbPost) -> int:
    return post.root_id or post.id


def record(s: Session, post: FbPost, kind: str, note: str | None = None, text: str | None = None) -> bool:
    """Dopisuje wersje (bez duplikatu ostatniego tekstu tego posta). Wymaga zapisanego posta (post.id)."""
    text = post.text if text is None else text
    last = s.exec(select(FbPostVersion).where(FbPostVersion.post_id == post.id)
                  .order_by(col(FbPostVersion.id).desc())).first()
    if last and last.text == text and kind != "round":
        return False
    s.add(FbPostVersion(post_id=post.id, root_id=root_of(post), text=text or "", kind=kind, note=note))
    return True


def ensure_initial(s: Session, post: FbPost) -> None:
    """Post sprzed historii: zapamietaj jego obecny tekst jako pierwsza wersje."""
    has = s.exec(select(FbPostVersion.id).where(FbPostVersion.post_id == post.id)).first()
    if not has and post.text:
        s.add(FbPostVersion(post_id=post.id, root_id=root_of(post), text=post.text,
                            kind="generated" if not post.edited else "initial",
                            created_at=post.created_at))


def family(s: Session, post_id: int) -> tuple[list[FbPost], list[FbPostVersion]]:
    post = s.get(FbPost, post_id)
    if not post:
        return [], []
    root = root_of(post)
    posts = s.exec(select(FbPost).where((FbPost.root_id == root) | (FbPost.id == root))
                   .order_by(FbPost.id)).all()
    for p in posts:
        ensure_initial(s, p)
    s.commit()
    versions = s.exec(select(FbPostVersion).where(FbPostVersion.root_id == root)
                      .order_by(FbPostVersion.created_at, FbPostVersion.id)).all()
    return posts, versions


def reviews_by_hash(s: Session, post_ids: list[int]) -> dict[str, FbPostReview]:
    out = {}
    for r in s.exec(select(FbPostReview).where(col(FbPostReview.post_id).in_(post_ids))
                    .order_by(FbPostReview.id)).all():
        out[r.text_hash] = r
    return out


def word_diff(a: str, b: str) -> dict:
    """Porownanie slowo po slowie: [(op, tekst)] op = eq | del | ins, plus liczby zmian."""
    ta, tb = _TOKEN.findall(a or ""), _TOKEN.findall(b or "")
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    ops, added, removed = [], 0, 0
    words = lambda xs: sum(1 for x in xs if x.strip() and re.search(r"\w", x))  # noqa: E731
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(("eq", "".join(ta[i1:i2])))
            continue
        if tag in ("replace", "delete"):
            ops.append(("del", "".join(ta[i1:i2])))
            removed += words(ta[i1:i2])
        if tag in ("replace", "insert"):
            ops.append(("ins", "".join(tb[j1:j2])))
            added += words(tb[j1:j2])
    return {"ops": ops, "added": added, "removed": removed, "similarity": round(100 * sm.ratio()),
            "same": a == b}


__all__ = ["KIND_LABELS", "record", "ensure_initial", "family", "reviews_by_hash", "word_diff", "text_hash"]
