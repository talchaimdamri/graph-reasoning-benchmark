// Hebrew dictionary for the entire UI. Keys are English identifiers; values are
// the Hebrew strings rendered to the user. Data values (model names, encoding
// names, graph node labels) are NOT translated.

export const he = {
  appTitle: "מדד חשיבה על גרפים",
  appSubtitle: "כמה טוב מודלי שפה מסיקים מסקנות על גרפים, לרוחב פורמטים שונים",
  generatedAt: "נוצר בתאריך",

  nav: {
    overview: "סקירה כללית",
    experiments: "ניסויים",
    pipeline: "שלבי הצנרת",
    explorer: "חוקר גרפים",
    metrics: "מדדים: מודלים מול פורמטים",
  },

  summary: {
    totalResults: "סה\"כ תוצאות",
    overallAccuracy: "דיוק כולל",
    models: "מודלים",
    encodings: "פורמטים",
    graphs: "גרפים",
    totalTokens: "סה\"כ טוקנים",
    errorRate: "שיעור שגיאות",
  },

  overview: {
    leaderboard: "טבלת מובילים — לפי מודל",
    encodingComparison: "השוואת פורמטים",
    accuracyByDifficulty: "דיוק לפי רמת קושי",
    accuracyByTier: "דיוק לפי גודל גרף",
    tokenEfficiency: "יעילות טוקנים (דיוק לכל 1000 טוקנים)",
    accuracyVsTokens: "דיוק מול עלות טוקנים",
  },

  experiments: {
    title: "כל הניסויים",
    description: "כל שורת תוצאה מהרצת המדד. ניתן לסנן לפי מודל, פורמט ורמת קושי.",
    filterModel: "מודל",
    filterEncoding: "פורמט",
    filterDifficulty: "רמת קושי",
    filterCorrect: "תקינות",
    all: "הכול",
    onlyCorrect: "תשובות נכונות",
    onlyWrong: "תשובות שגויות",
    showing: "מציג",
    of: "מתוך",
    rows: "שורות",
    columns: {
      graph: "גרף",
      encoding: "פורמט",
      model: "מודל",
      question: "שאלה",
      difficulty: "קושי",
      category: "קטגוריה",
      correct: "נכון",
      tokens: "טוקנים",
      latency: "השהיה (ms)",
      error: "שגיאה",
    },
    yes: "כן",
    no: "לא",
    none: "—",
  },

  pipeline: {
    title: "שלבי הצנרת",
    description:
      "המדד בנוי כצנרת בת חמישה שלבים. כל שלב מזין את הבא, מיצירת הגרפים ועד ניתוח התוצאות.",
    stages: [
      {
        key: "generate",
        title: "יצירה",
        body: "יצירת גרפים סינתטיים בשלוש דרגות גודל (קטן, בינוני, גדול) עם זרעים אקראיים קבועים לשחזוריות. כוללים גרפים אקראיים, היררכיים וחסרי-קנה-מידה.",
      },
      {
        key: "encode",
        title: "קידוד",
        body: "המרת כל גרף למספר פורמטים טקסטואליים: רשימת שכנויות, רשימת קשתות, Mermaid, DOT, שפה טבעית ומטריצה — וכן ייצוג ויזואלי.",
      },
      {
        key: "question",
        title: "שאלות",
        body: "יצירת שאלות דטרמיניסטיות מבוססות-תבנית לכל גרף, עם תשובות אמת מחושבות באמצעות NetworkX. השאלות מסווגות לפי קטגוריה ורמת קושי.",
      },
      {
        key: "benchmark",
        title: "הרצת המדד",
        body: "הרצת כל תא בגריד (גרף × פורמט × שאלה × מודל) מול מודל השפה, עם מטמון ב-SQLite למניעת חישוב כפול ולהמשך הרצה לאחר עצירה.",
      },
      {
        key: "analyze",
        title: "ניתוח",
        body: "צבירת התוצאות לטבלאות מדדים, יצירת גרפים באיכות פרסום וייצוא קובץ results.json שלוח הבקרה הזה קורא.",
      },
    ],
  },

  explorer: {
    title: "חוקר גרפים אינטראקטיבי",
    description:
      "בחרו גרף כדי לראות את המבנה שלו. ניתן לגרור צמתים, לקרב ולהרחיק. הצמתים והקשתות מגיעים ישירות מהגרפים שנוצרו במדד.",
    selectGraph: "בחירת גרף",
    nodes: "צמתים",
    edges: "קשתות",
    tier: "דרגה",
    directed: "מכוון",
    weighted: "משוקלל",
    layout: "פריסה",
  },

  metricsView: {
    title: "מדדים: מודלים מול פורמטים",
    description: "מפת חום של דיוק עבור כל צירוף של מודל ופורמט קידוד.",
    heatmap: "מפת חום — דיוק",
    table: "טבלת דיוק",
    model: "מודל \\ פורמט",
    byCategory: "דיוק לפי קטגוריית שאלה",
    errorBreakdown: "פילוח סוגי שגיאות",
  },

  chart: {
    accuracy: "דיוק",
    tokens: "טוקנים",
    count: "כמות",
    meanTokens: "ממוצע טוקנים",
    accuracyPer1k: "דיוק לכל 1000 טוקנים",
  },

  loading: "טוען נתונים…",
  loadError: "טעינת results.json נכשלה. הריצו את scripts/build_dashboard_data.py.",
  noData: "אין נתונים להצגה.",
};

export type Dict = typeof he;
