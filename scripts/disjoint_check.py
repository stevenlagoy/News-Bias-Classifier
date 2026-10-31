from partisan_classifier.data.loader import load_media_split

tr, va, te = load_media_split("data/raw")
shared = set(tr["publisher"]) & set(te["publisher"])
for p in sorted(shared):
    print(p, (tr["publisher"] == p).sum(), (te["publisher"] == p).sum(),
          tr.loc[tr["publisher"] == p, "label"].unique(),
          te.loc[te["publisher"] == p, "label"].unique())
print("share of test articles from publishers seen in train:", te["publisher"].isin(shared).mean())