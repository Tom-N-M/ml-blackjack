# Wiederverwendbare Projektdokumentation

Die Klasse `bhhproject.cls` bündelt das Dokumentlayout, die Titelseite,
PDF-Metadaten, Kopf- und Fußzeilen sowie die Formatierung für Tabellen,
Abbildungen und Quellcode.

Für ein neues Projekt werden die Klasse und eine kurze Hauptdatei benötigt:

```tex
\documentclass{bhhproject}

\projectmodule{Machine Learning}
\projectsemester{Sommersemester 2026}
\projecttype{Projektdokumentation}
\projecttitle{Titel des Projekts}
\projectshorttitle{Kurztitel für die Kopfzeile}
\projectsubtitle{Optionaler Untertitel}
\projectpdfauthors{Vorname Nachname, Vorname Nachname}
\projectplace{Hamburg}
\projectdate{16. Juni 2026}

\addprojectauthor
  {Vorname Nachname}
  {1234567}
  {Musterstraße 1, 20095 Hamburg}
  {vorname.nachname@example.com}

\addprojectauthor
  {Zweite Person}
  {7654321}
  {Beispielweg 2, 20095 Hamburg}
  {zweite.person@example.com}

\begin{document}
\makeprojecttitle
% Inhalt
\end{document}

```

`\addprojectauthor` kann beliebig oft aufgerufen werden. Zwei bis vier
Beteiligte passen im vorgesehenen Tabellenlayout auf die Titelseite. Für jede
Person werden Name, Matrikelnummer, Anschrift und E-Mail-Adresse ausgegeben.