# GRB Dashboard — לוח בקרה למדד חשיבה על גרפים

לוח בקרה מקומי (Local) ב-React + TypeScript להצגת תוצאות ה-Graph Reasoning Benchmark.
ה-UI כולו בעברית (RTL), בעוד שכל הנתונים, שמות המודלים, פורמטי הקידוד ותוויות הצמתים נשארים באנגלית.

A local React + TypeScript dashboard for the Graph Reasoning Benchmark.
The UI is entirely Hebrew (RTL); all data, model names, encoding formats and graph node
labels stay in English.

---

## הרצה / Running

```bash
cd dashboard
npm install
npm run dev      # פיתוח / dev server at http://localhost:5173
```

בנייה לפרודקשן / Production build:

```bash
npm run build    # מייצר תיקיית dist/ ; חייב לעבור
npm run preview  # תצוגה מקדימה של הבנייה
```

---

## נתונים / Data

הלוח קורא את הקובץ `public/results.json`. ליצירתו מחדש (כולל הגרפים באיכות פרסום
בתיקיית `figures/`) הריצו מהשורש של הפרויקט:

The dashboard reads `public/results.json`. To regenerate it (and the
publication-quality figures under `figures/`), run from the repo root:

```bash
# נתוני דמו סינתטיים / synthetic demo data (default)
python scripts/build_dashboard_data.py

# מתוצאות אמת ב-SQLite / from a real SQLite results DB
python scripts/build_dashboard_data.py --db path/to/results.sqlite
```

אם `results.json` לא קיים, הלוח יראה הודעת שגיאה בעברית.
A synthetic fixture ships in `public/results.json` so the dashboard renders before any real run.

---

## תצוגות / Views

1. **סקירה כללית (Overview)** — טבלת מובילים לפי מודל, השוואת פורמטים, יעילות טוקנים ודיוק מול עלות.
2. **מדדים: מודלים מול פורמטים (Metrics)** — מפת חום של דיוק, דיוק לפי קטגוריה ופילוח שגיאות.
3. **ניסויים (Experiments)** — כל שורת תוצאה, עם סינון לפי מודל / פורמט / רמת קושי / תקינות.
4. **חוקר גרפים (Explorer)** — הצגה אינטראקטיבית (Cytoscape.js) של הגרפים שנוצרו.
5. **שלבי הצנרת (Pipeline)** — הסבר בעברית על חמשת שלבי המדד: generate → encode → question → benchmark → analyze.

---

## מחסנית טכנולוגית / Stack

- React 19 + TypeScript + Vite
- [Cytoscape.js](https://js.cytoscape.org/) — interactive graph visualization
- [Recharts](https://recharts.org/) — charts
- מילון i18n בעברית ב-`src/i18n/he.ts`
