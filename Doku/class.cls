\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{class}[2025/11/09 BHH Bachelorthesis LaTex class]

%\LoadClass[12pt, ngerman, a4paper, toc=left, parskip=half, numbers=noenddot, headings=standardclasses, nonumberlist, singlespacing=true]{scrartcl}
\LoadClass[12pt, ngerman, a4paper, numbers=noenddot, singlespacing=true, runningheads]{llncs}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[ngerman]{babel}
\RequirePackage[ngerman=ngerman-x-latest]{hyphsubst}

%\usepackage[a4paper, top=2.5cm, right=2.5cm, bottom=2cm, left=2.5cm, footskip=0.5cm, head=21.75pt]{geometry}
\usepackage[a4paper, top=5cm, right=4.5cm, bottom=5.5cm, left=4.5cm, footskip=0.5cm, head=21.75pt]{geometry}
\usepackage{afterpage}
\usepackage{calc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{tabularx}
\usepackage{tabulary}
% \usepackage{ltablex}
\usepackage{booktabs}
\usepackage[official]{eurosym}
\usepackage{lastpage}
\usepackage{etoolbox}
\usepackage{amssymb}
\usepackage{enumitem}

\usepackage[numbers,sort&compress]{natbib}

\usepackage[hidelinks]{hyperref}
\usepackage{dramatist}
\usepackage{listings}

% table addons
\usepackage[table]{xcolor}
\usepackage{multirow}
\usepackage{array}

\usepackage{adjustbox,lipsum}
\usepackage{makecell}

\usepackage{outlines}

% für nicht Fettgedruckte Überschriften auskommentieren:
% \setkomafont{disposition}{\normalcolor\normalfont}

%---------------- Glossar
\usepackage[acronym, toc]{glossaries}
% ------------- Abbildungen mit Tikz erstellen
\usepackage{tikz}
\usetikzlibrary{arrows,automata,shapes,calc}


% ------------- Zeilenabstand
\usepackage{setspace}
%\renewcommand{\baselinestretch}{1.5}

\makeatletter
\@twosidefalse
\@mparswitchfalse
\makeatother

\bibliographystyle{ieeetr}
\lstset{
	breakatwhitespace=false,         % sets if automatic breaks should only happen at whitespace
	breaklines=true,                 % sets automatic line breaking
	frame=single,                    % adds a frame around the code
	keepspaces=true,                 % keeps spaces in text, useful for keeping indentation of code (possibly needs columns=flexible)
	numbers=left,                    % where to put the line-numbers; possible values are (none, left, right)
	numbersep=5pt,                   % how far the line-numbers are from the code
	numberstyle=\color{gray},        % the style that is used for the line-numbers
	rulecolor=\color{black},         % if not set, the frame-color may be changed on line-breaks within not-black text (e.g. comments (green here))
}


\renewcommand{\contentsname}{Inhaltsverzeichnis}
\renewcommand{\listfigurename}{Abbildungsverzeichnis}
\renewcommand{\listtablename}{Tabellenverzeichnis}
\renewcommand{\refname}{Literaturverzeichnis}

\usepackage[automark]{scrlayer-scrpage}
\pagestyle{scrheadings}
\clearmainofpairofpagestyles
\ihead{}
\chead{}
\ofoot{\pagemark}

% variables for the title
\newcommand{\fieldofstudy}[1]{\renewcommand{\fieldofstudy}{#1}}
\newcommand{\firstexaminer}[1]{\renewcommand{\firstexaminer}{#1}}
\newcommand{\secondexaminer}[1]{\renewcommand{\secondexaminer}{#1}}
\newcommand{\deadline}[1]{\renewcommand{\deadline}{#1}}
\newcommand{\semester}[1]{\renewcommand{\semester}{#1}}
\newcommand{\semesternumber}[1]{\renewcommand{\semesternumber}{#1}}
\newcommand{\dateofbirth}[1]{\renewcommand{\dateofbirth}{#1}}
\newcommand{\registrationnumber}[1]{\renewcommand{\registrationnumber}{#1}}
\newcommand{\address}[1]{\renewcommand{\address}{#1}}
\newcommand{\studemail}[1]{\renewcommand{\studemail}{#1}}
\newcommand{\worktitle}[1]{\renewcommand{\worktitle}{#1}}
\newcommand{\worktitleeid}[1]{\renewcommand{\worktitleeid}{#1}}
\newcommand{\studname}[1]{\renewcommand{\studname}{#1}}

% title page layouts
\renewcommand*{\maketitle}{%
	\begin{titlepage}
	\hypersetup{pageanchor=false}
	\thispagestyle{empty}
	\newgeometry{top=2.5cm, right=2.5cm, bottom=2cm, left=2.5cm, footskip=0.5cm, head=21.75pt}
		\vspace{1cm}
		\begin{center}
			\vspace{3cm}
			{\large Berufliche Hochschule Hamburg} \\
			\vspace{1cm}
			{\includegraphics[width=5cm,height=10cm,keepaspectratio]{ba/images/bhhlogo.png}}\\
			\vspace{1cm}
			\begin{spacing}{2}
				{\huge \textbf{\worktitle}}\\
			\end{spacing}
			\vspace{1.5cm}
			{\large zur Erreichung des akademischen Grades}\\
			\vspace{0.5cm}
			{\huge Bachelor of Science} \\
			\vspace{0.5cm}
			{\large des Studiengangs}\\
			\vspace{0.5cm}
			{\huge Informatik}\\
		\end{center}

		\vspace{0.5cm}

		\begin{flushleft}
			\textbf{Erster Gutachter:} \\
			\firstexaminer\\
			\textbf{Zweiter Gutachter:} \\
			\secondexaminer\\
			\textbf{Abgabe:} \\
			Hamburg, der \deadline

			\vspace{0.5cm}

			\textbf{Vorgelegt von:} \\
			\begin{tabular}{ll}
				Name:           & \studname           \\
				Geburtsdatum:   & \dateofbirth        \\
				Fachsemester:   & \semesternumber     \\
				Matrikelnummer: & \registrationnumber \\
				Adresse:        & \address            \\
				E-Mail:         & \studemail          \\
			\end{tabular}
		\end{flushleft}
		%
	\end{titlepage}
	\restoregeometry
}

\newcommand{\makeeidesstattlicheerklaerung}{%
    \section*{Eidesstattliche Versicherung}
    Hiermit erkläre ich, \textbf{\studname}, geboren am \textbf{\dateofbirth}, Matrikel Nummer \textbf{\registrationnumber}, an Eides Statt, dass ich die Bachelorarbeit mit dem Titel \\

    \textbf{\worktitle}\\

    selbst, ohne Hilfe Dritter und ausschließlich unter Verwendung der angegebenen Quellen angefertigt habe. Alle Stellen, die wörtlich oder sinngemäß aus veröffentlichten oder nicht veröffentlichten Quellen entnommen sind, habe ich als solche kenntlich gemacht und alle Quellen, die dem Internet entnommen oder in einer sonstigen digitalen Form verwendet wurden, sind der Arbeit beigefügt. Die Arbeit wurde bisher in gleicher oder ähnlicher Form, auch nicht in Teilen, keiner anderen Prüfungsbehörde vorgelegt und auch nicht veröffentlicht.\\

    Sofern generative KI-Tools verwendet wurden, habe ich Produktnamen, Hersteller, die jeweils verwendete Softwareversion und die jeweiligen Einsatzzwecke (z.B. sprachliche Überprüfung und Verbesserung der Texte, systematische Recherche) benannt. Ich verantworte die Auswahl, die Übernahme und sämtliche Ergebnisse des von mir verwendeten KI-generierten Outputs vollumfänglich selbst.\\

    Die über das BHH-Studierendenportal eingereichte Fassung entspricht der abgegebenen gedruckten Fassung.\\

    Ich bin mir bewusst, dass eine falsche eidesstattliche Versicherung strafrechtliche Folgen haben kann.\\

    \begin{table*}
        \begin{tabular}{p{3cm} p{0.5cm} p{3cm} p{0.5cm} p{3cm}}
            &&&&\\
            \textbf{Hamburg}&&\textbf{24. April 2026}&&\\
            \cline{1-1}
            \cline{3-3}
            \cline{5-5}
            Ort & & Datum && Unterschrift  \\
        \end{tabular}
    \end{table*}
}
