\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bhhproject}[2026/06/13 Reusable BHH project documentation class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrartcl}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper,bibliography=totoc,listof=totoc]{scrartcl}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[ngerman]{babel}
\RequirePackage{lmodern}
\RequirePackage{substitutefont}
\RequirePackage{microtype}
\RequirePackage[left=30mm,right=25mm,top=25mm,bottom=25mm]{geometry}
\RequirePackage{setspace}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage[labelfont=bf]{caption}
\RequirePackage{xcolor}
\RequirePackage{colortbl}
\RequirePackage{listings}
\RequirePackage{etoolbox}
\RequirePackage{scrlayer-scrpage}
\RequirePackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}

\definecolor{BHHGray}{HTML}{64748B}
\definecolor{codegreen}{rgb}{0,0.5,0}
\definecolor{codegray}{rgb}{0.45,0.45,0.45}
\definecolor{backcolour}{rgb}{0.96,0.96,0.94}

\onehalfspacing
\setlength{\emergencystretch}{2em}
\setcounter{tocdepth}{2}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showstringspaces=false,
    tabsize=2
}
\lstset{style=mystyle}

\clearpairofpagestyles
\ihead{\small\color{BHHGray}\projectshorttitlevalue}
\ofoot{\pagemark}
\setkomafont{pageheadfoot}{\normalfont}
\pagestyle{scrheadings}

\newcommand{\projecttitlevalue}{Projektdokumentation}
\newcommand{\projectshorttitlevalue}{Projektdokumentation}
\newcommand{\projectsubtitlevalue}{}
\newcommand{\projectsemestervalue}{}
\newcommand{\projectplacevalue}{}
\newcommand{\projectdatevalue}{\today}
\newcommand{\projectauthorsvalue}{}
\newcommand{\projectpdfauthorsvalue}{}
\newcommand{\projecttypevalue}{Projektdokumentation}
\newcommand{\projectmodulevalue}{}

\newcommand{\projectauthorrows}{}
\newcounter{projectauthorcount}

% Öffentliche User-Befehle
\newcommand{\projecttitle}[1]{\renewcommand{\projecttitlevalue}{#1}}
\newcommand{\projectshorttitle}[1]{\renewcommand{\projectshorttitlevalue}{#1}}
\newcommand{\projectsubtitle}[1]{\renewcommand{\projectsubtitlevalue}{#1}}
\newcommand{\projectsemester}[1]{\renewcommand{\projectsemestervalue}{#1}}
\newcommand{\projectplace}[1]{\renewcommand{\projectplacevalue}{#1}}
\newcommand{\projectdate}[1]{\renewcommand{\projectdatevalue}{#1}}
\newcommand{\projectauthors}[1]{\renewcommand{\projectauthorsvalue}{#1}}
\newcommand{\projectpdfauthors}[1]{\renewcommand{\projectpdfauthorsvalue}{#1}}
\newcommand{\projecttype}[1]{\renewcommand{\projecttypevalue}{#1}}
\newcommand{\projectmodule}[1]{\renewcommand{\projectmodulevalue}{#1}}

% Hilfsfunktionen für die Autorentabelle
\newcommand{\projectmissing}{\textit{\color{BHHGray}bitte ergänzen}}
\newcommand{\projectfield}[1]{\ifstrempty{#1}{\projectmissing}{#1}}
\newcommand{\projectemailfield}[1]{\ifstrempty{#1}{\projectmissing}{\href{mailto:#1}{#1}}}
\newcommand{\addprojectauthor}[4]{%
    \stepcounter{projectauthorcount}%
    \gappto{\projectauthorrows}{#1 & \projectfield{#2} & \projectfield{#3} & \projectemailfield{#4}\\}%
}

\newcommand{\projectauthorsblock}{%
    \ifnum\value{projectauthorcount}>0
        {\footnotesize
        \noindent\textbf{Vorgelegt von:}\par
        \vspace{0.08cm}
        \renewcommand{\arraystretch}{1.12}%
        \begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{28mm}>{\raggedright\arraybackslash}p{18mm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{40mm}@{}}
            \textbf{Name} & \textbf{Matr.-Nr.} & \textbf{Anschrift} & \textbf{E-Mail}\\
            \projectauthorrows
        \end{tabularx}\par}
    \else
        % Falls keine tabellarischen Autoren da sind, wird der Freitext genutzt
        \ifdefempty{\projectauthorsvalue}{}{\noindent\textbf{Projektgruppe:} \projectauthorsvalue\par}
    \fi
}

\newcommand{\makeprojecttitle}{%
    \hypersetup{pageanchor=false}%

    \begin{titlepage}
        \newgeometry{top=25mm,right=25mm,bottom=20mm,left=25mm,footskip=5mm,headheight=21.75pt}
        \thispagestyle{empty}
        \begin{center}
	        {\large Berufliche Hochschule Hamburg} \\
        \vspace{1cm}
        {\includegraphics[width=5cm,height=10cm,keepaspectratio]{Images/bhhlogo.png}}\\
        \vspace{1cm}
            \huge Studiengang \\
            \vspace{1cm}
            {\huge Informatik B.Sc.} \\
            \vspace{2cm}
            {\huge \textbf{\projecttitlevalue}} \\
            \vspace{1cm}
            {\Large \projectsubtitlevalue} \\
            \vspace{1cm}
            {\large \projectsemestervalue} \\
        \end{center}
        \vspace{2cm}

        \begin{flushleft}
            Abgabe: \\
            \projectplacevalue, den \projectdatevalue

            \vspace{1cm}

            \projectauthorsblock
        \end{flushleft}
        \restoregeometry
    \end{titlepage}
    \hypersetup{pageanchor=true}%
}

\AtBeginDocument{%
    \hypersetup{
        pdftitle={\projecttitlevalue},
        pdfauthor={\projectpdfauthorsvalue},
        pdfsubject={\projecttypevalue\space -- \projectmodulevalue}
    }%
}
