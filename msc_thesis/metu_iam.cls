\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{metu_iam}[2003/06/20 METU Thesis Class]
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
% Use Times New Roman
\RequirePackage{times}

%default options

%spacing - default double
\newif\ifd@uble \d@ublefalse
\newif\ifs@ngle \s@ngletrue

%page numbering - default 
\newif\ifpn@top \pn@topfalse
\newif\ifpn@topr \pn@toprtrue

%chapter numbering (roman/arabic)
\newif\if@rmnchp \@rmnchptrue

%table stuff
\newif\ifmy@t@ble \my@t@blefalse
\newif\ifmy@figur@ \my@figur@false
\newif\ifmy@t@ble \my@t@blefalse

%font size
\newif\if@tenpt \@tenptfalse
\newif\if@elevenpt \@elevenpttrue
\newif\if@twelvept \@twelveptfalse

% seems we need it.
\newif\if@twoside \@twosidetrue 
\@mparswitchtrue \def\ds@draft{\overfullrule 5pt}

\newif\ift@rkish \t@rkishfalse

%Font Options
\DeclareOption{10pt}{\@tenpttrue \@elevenptfalse \@twelveptfalse}
\DeclareOption{11pt}{\@tenptfalse \@elevenpttrue \@twelveptfalse}
\DeclareOption{12pt}{\@tenptfalse \@elevenptfalse \@twelvepttrue}

%Page Numbering
\DeclareOption{pntr}{\pn@toptrue \pn@toprtrue}
\DeclareOption{pntc}{\pn@toptrue \pn@toprfalse}
\DeclareOption{pnbt}{\pn@topfalse \pn@toprfalse}

%Chapter Numbering
\DeclareOption{chaproman}{\@rmnchptrue}
\DeclareOption{chaparabic}{\@rmnchpfalse}

%Spacing
\DeclareOption{double}{\d@ubletrue \s@nglefalse}
\DeclareOption{oneandhalf}{\d@ublefalse \s@nglefalse}
\DeclareOption{single}{\d@ublefalse \s@ngletrue}

%print style
\DeclareOption{twoside}{\@twosidetrue}
\DeclareOption{oneside}{\@twosidefalse}

%Language
\DeclareOption{tr}{\t@rkishtrue}
\DeclareOption{eng}{\t@rkishfalse}

\def\@deptacsc{Actuarial Sciences} \def\@tdeptacsc{Aktüerya Bilimleri}
\def\@deptcryp{Cryptography} \def\@tdeptcryp{Kriptografi}
\def\@deptfm{Financial Mathematics} \def\@tdeptfm{Finansal Matematik}
\def\@deptsc{Scientific Computing} \def\@tdeptsc{Bilimsel Hesaplama}
\def\@deptaee{Aerospace Engineering} \def\@tdeptaee{Havacılık ve Uzay Mühendisliği}
\def\@deptarme{Archaeometry} \def\@tdeptarme{Arkeometri}
\def\@deptarch{Architecture} \def\@tdeptarch{Mimarlık}
\def\@deptbch{Biochemistry} \def\@tdeptbch{Biyokimya}
\def\@deptbiol{Biology} \def\@tdeptbiol{Biyoloji}
\def\@deptbme{Biomedical Engineering} \def\@tdeptbme{Biyomedikal Mühendisliği}
\def\@deptbtec{Biotechnology} \def\@tdeptbtec{Biyoteknoloji}
\def\@deptbs{Building Science} \def\@tdeptbs{Yapı Bilimleri}
\def\@deptceme{Cement Engineering} \def\@tdeptceme{Çimento Mühendisliği}
\def\@deptche{Chemical Engineering} \def\@tdeptche{Kimya Mühendisliği}
\def\@deptchem{Chemistry} \def\@tdeptchem{Kimya}
\def\@deptcrp{City and Regional Planning} \def\@tdeptcrp{Şehir ve Bölge Planlama}
\def\@deptcp{City Planning} \def\@tdeptcp{Şehir Planlama}
\def\@deptce{Civil Engineering} \def\@tdeptce{İnşaat Mühendisliği}
\def\@deptarcd{Computational Design and Fabrication Technologies in Architecture} \def\@tdeptarcd{Mimarlıkta Sayısal Tasarım ve Üretim Teknolojileri}
\def\@deptcte{Computer Education and Instructional Technology} \def\@tdeptcte{Bilgisayar ve Öğretim Teknolojileri Öğretmenliği}	
\def\@deptiddi{Design Research for Interaction} \def\@tdeptiddi{Etkileşim İçin Tasarım}
\def\@depteqs{Earthquake Studies} \def\@tdepteqs{Deprem Çalışmaları}
\def\@deptess{Earth System Science} \def\@tdeptess{Yer Sistemi Bilimleri}
\def\@deptee{Electrical and Electronics Engineering} \def\@tdeptee{Elektrik ve Elektronik Mühendisliği}
\def\@deptem{Engineering Management} \def\@tdeptem{Mühendislik Yönetimi}
\def\@deptes{Engineering Sciences} \def\@tdeptes{Mühendislik Bilimleri}
\def\@deptenve{Environmental Engineering} \def\@tdeptenve{Çevre Mühendisliği}
\def\@deptfde{Food Engineering} \def\@tdeptfde{Gıda Mühendisliği}
\def\@deptggit{Geodetics - Geographical Information Technologies} \def\@tdeptggit{Jeodezi ve Coğrafi Bilgi Teknolojileri}
\def\@deptgeoe{Geological Engineering} \def\@tdeptgeoe{Jeoloji Mühendisliği}
\def\@depthe{Hydrosystems Engineering} \def\@tdepthe{Hidrosistem Mühendisliği}
\def\@deptid{Industrial Design} \def\@tdeptid{Endüstri Ürünleri Tasarımı}
\def\@deptie{Industrial Engineering} \def\@tdeptie{Endüstri Mühendisliği}
\def\@deptmath{Mathematics} \def\@tdeptmath{Matematik}
\def\@deptmech{Mechanical Engineering} \def\@tdeptmech{Makina Mühendisliği}
\def\@deptmete{Metallurgical and Materials Engineering} \def\@tdeptmete{Metalurji ve Malzeme Mühendisliği}
\def\@deptmnt{Micro and Nanotechnology} \def\@tdeptmnt{Mikro ve Nanoteknoloji}
\def\@deptmine{Mining Engineering} \def\@tdeptmine{Maden Mühendisliği}
\def\@deptor{Operational Research} \def\@tdeptor{Yöneylem Araştırması}
\def\@deptpete{Petroleum and Natural Gas Engineering} \def\@tdeptpete{Petrol ve Doğalgaz Mühendisliği}
\def\@deptphys{Physics} \def\@tdeptphys{Fizik}
\def\@deptpst{Polymer Science and Technology} \def\@tdeptpst{Polimer Bilimi ve Teknolojisi}
\def\@deptrp{Regional Planning} \def\@tdeptrp{Bölge Planlama}
\def\@deptrest{Restoration} \def\@tdeptrest{Restorasyon}
\def\@deptssme{Secondary Science and Mathematics Education} \def\@tdeptssme{Ortaöğretim Fen ve Matematik Alan Eğitimi}
\def\@deptse{Software Engineering} \def\@tdeptse{Yazılım Mühendisliği}
\def\@deptstat{Statistics} \def\@tdeptstat{İstatistik}
\def\@deptst{Structural Mechanics} \def\@tdeptst{Yapı Mekaniği}
\def\@department{}
\def\@tdepartment{}
\def\@deptceng{Computer Engineering} \def\@tdeptceng{Bilgisayar Mühendisliği}
\def\DeclareDepartment#1{
	\def\@department{\csname @dept#1\endcsname}
	\def\@tdepartment{\csname @tdept#1\endcsname}
}
\DeclareOption{acsc}{\DeclareDepartment{acsc}}
\DeclareOption{cryp}{\DeclareDepartment{cryp}}
\DeclareOption{fm}{\DeclareDepartment{fm}}
\DeclareOption{sc}{\DeclareDepartment{sc}}
\DeclareOption{ceng}{\DeclareDepartment{ceng}}
\DeclareOption{aee}{\DeclareDepartment{aeee}}
\DeclareOption{arme}{\DeclareDepartment{arme}}
\DeclareOption{arch}{\DeclareDepartment{arch}}
\DeclareOption{bch}{\DeclareDepartment{bch}}
\DeclareOption{biol}{\DeclareDepartment{biol}}
\DeclareOption{bme}{\DeclareDepartment{bme}}
\DeclareOption{btec}{\DeclareDepartment{btec}}
\DeclareOption{bs}{\DeclareDepartment{bs}}
\DeclareOption{ceme}{\DeclareDepartment{ceme}}
\DeclareOption{che}{\DeclareDepartment{che}}
\DeclareOption{chem}{\DeclareDepartment{chem}}
\DeclareOption{crp}{\DeclareDepartment{crp}}
\DeclareOption{cp}{\DeclareDepartment{cp}}
\DeclareOption{ce}{\DeclareDepartment{ce}}
\DeclareOption{arcd}{\DeclareDepartment{arcd}}
\DeclareOption{cte}{\DeclareDepartment{cte}}
\DeclareOption{iddi}{\DeclareDepartment{iddi}}
\DeclareOption{eqs}{\DeclareDepartment{eqs}}
\DeclareOption{ess}{\DeclareDepartment{ess}}
\DeclareOption{ee}{\DeclareDepartment{ee}}
\DeclareOption{em}{\DeclareDepartment{em}}
\DeclareOption{es}{\DeclareDepartment{es}}
\DeclareOption{enve}{\DeclareDepartment{enve}}
\DeclareOption{fde}{\DeclareDepartment{fde}}
\DeclareOption{ggit}{\DeclareDepartment{ggit}}
\DeclareOption{geoe}{\DeclareDepartment{geoe}}
\DeclareOption{he}{\DeclareDepartment{he}}
\DeclareOption{id}{\DeclareDepartment{id}}
\DeclareOption{ie}{\DeclareDepartment{ie}}
\DeclareOption{math}{\DeclareDepartment{math}}
\DeclareOption{mech}{\DeclareDepartment{mech}}
\DeclareOption{mete}{\DeclareDepartment{mete}}
\DeclareOption{mnt}{\DeclareDepartment{mnt}}
\DeclareOption{mine}{\DeclareDepartment{mine}}
\DeclareOption{or}{\DeclareDepartment{or}}
\DeclareOption{pete}{\DeclareDepartment{pete}}
\DeclareOption{phys}{\DeclareDepartment{phys}}
\DeclareOption{pst}{\DeclareDepartment{pst}}
\DeclareOption{rp}{\DeclareDepartment{rp}}
\DeclareOption{rest}{\DeclareDepartment{rest}}
\DeclareOption{ssme}{\DeclareDepartment{ssme}}
\DeclareOption{se}{\DeclareDepartment{se}}
\DeclareOption{stat}{\DeclareDepartment{stat}}
\DeclareOption{st}{\DeclareDepartment{st}}

% Phd or MS
\newif\ifp@hd \p@hdfalse
\DeclareOption{phd}{\p@hdtrue}
\DeclareOption{ms}{\p@hdfalse}

%Unspecified input handler
\DeclareOption*{Invalid Option \CurrentOption }

\ProcessOptions\relax
%\LoadClass{book}

\newif\ifturkish
\ift@rkish
\RequirePackage[english,turkish]{babel}
\turkishtrue 
\else
\RequirePackage[turkish,english]{babel}
\turkishfalse
\fi


\addto\extrasturkish{%
  \uccode`i=\string"9D \uccode\string"19 = `I
  \lccode\string"9D=`i \lccode`I=\string"19 }
\addto\noextrasturkish{%
  \uccode`i=`I \lccode`I=`i }
\ift@rkish
%%% TURKISH Text Templates

\def\@tocl{İÇİNDEKİLER}
\def\@lot{TABLOLAR LİSTESİ}
\def\@lof{ŞEKİLLER LİSTESİ}
\def\@loab{KISALTMALAR LİSTESİ}
\def\@chapterslabel{BÖLÜMLER}
\def\@figureslabel{ŞEKİLLER}
\def\@tableslabel{TABLOLAR}
\def\@tablelabel{Tablo}
\def\@titleparta{ORTA DOĞU TEKNİK ÜNİVERSİTESİ\\
       \MakeUppercase{\expandafter{\@tgraduateschool}}\\
	\MakeUppercase{\@tdegree} TEZİ}
\def\@titlepartb{\MakeUppercase{\@tdepartment}\\
	ANA BİLİM DALI}
\def\@approvallabel{Tezin Onayı:}
\def\@approvaltext{{\bf \MakeUppercase{\@author}} tarafından, {\bf Orta Doğu Teknik Üniversitesi, \@tdepartment} ana bilim dalında {\bf \@tdegree} derecesinin gerekliliklerini kısmen yerine getirmek için teslim edilmiştir,}
\def\@directoraff{Enstitü Müdürü, {\bf Fen Bilimleri Enstitüsü}}
\def\@chairlabel{Bölüm Başkanı}
\def\@supervisorlabel{Tez Yöneticisi}
\def\@cosupervisorlabel{Ortak Tez Yöneticisi}
\def\@committeelabel{Juri Üyeleri}
\def\@datelabel{Tarih}
\def\@plagirismtext{Bu metindeki tüm bilgilerin akademik kurallar ve ahlaki prensipler çerçevesinde
elde edildiğini ve sunulduğunu teyit ederim. Ayrıca kuralların ve prensiplerin gerektirdiği gibi, bu
metinde orjinal olmayan tüm materyal alıntı olarak belirtilmiş ve gerekli referanslar verilmiştir.}
\def\@plagirismnametext{Ad, Soyad}
\def\@plagirismsignaturetext{İmza}
\def\@chapapp{\vskip4mm BÖLÜM}

\else
%%% ENGLISH Text Templates
\def\@tocl{TABLE OF CONTENTS}
\def\@lot{LIST OF TABLES}
\def\@lof{LIST OF FIGURES}
\def\@loab{LIST OF ABBREVIATIONS}
\def\@chapterslabel{CHAPTERS}
\def\@figureslabel{FIGURES}
\def\@tableslabel{TABLES}
\def\@tablelabel{Table}
\def\@titleparta{A THESIS SUBMITTED TO\\
       THE GRADUATE SCHOOL OF \uppercase\expandafter{\@graduateschool}\\
       OF\\ MIDDLE EAST TECHNICAL UNIVERSITY}
\def\@titleseperator{BY}
\def\@titlepartb{IN PARTIAL FULFILLMENT OF THE REQUIREMENTS\\
       FOR\\ 
       THE DEGREE OF \uppercase\expandafter{\@degree}\\
       IN\\
       \MakeUppercase{\@department}}
\def\@approvallabel{Approval of the thesis:}
\def\@approvaltext{submitted by {\bf \MakeUppercase{\@author}} in partial fulfillment of the
requirements for the degree of {\bf \@degree{} in \@department{} Department, Middle East
Technical University} by,}
\def\@directoraff{Director, Graduate School of {\bf Applied Mathematics}}
\def\@chairlabel{Head of Department}
\def\@supervisorlabel{Supervisor}
\def\@cosupervisorlabel{Co-supervisor}
\def\@committeelabel{Examining Committee Members}
\def\@datelabel{Date}
\def\@plagirismtext{I hereby declare that all information in this document has been obtained and
presented in accordance with academic rules and ethical conduct. I also declare that, as required by
these rules and conduct, I have fully cited and referenced all material and results that are not
original to this work.}
\def\@plagirismnametext{Name, Last Name}
\def\@plagirismsignaturetext{Signature}
\def\@chapapp{\vskip4mm CHAPTER}
\fi

% Font naming
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}

%input the appropriate .def file
\if@tenpt \input{metu10.def} \fi
\if@elevenpt \input{metu11.def} \fi
\if@twelvept \input{metu12.def} \fi

\newif\if@lg@rithmloaded \@lg@rithmloadedfalse
\AtBeginDocument{
\@ifpackageloaded{algorithm2e}{\@lg@rithmloadedtrue}{\@lg@rithmloadedfalse}
}

%\ifpn@top\footheight=0cm\footskip=0cm%
%\else\footheight=12pt\footskip=30pt\headsep=0cm\headheight=0cm\fi

\ifpn@top
    \footskip=0cm%
\else
    \footskip=30pt\headsep=0cm\headheight=0cm
\fi

%\def\expthr@@#1{%
%   \expandafter\expandafter\expandafter%
%   \uplowcase\expandafter{#1}}
%\def\expthr@@#1{ \expandafter\expandafter\expandafter\uplowcase{#1}}
\def\expthr@@#1{#1}

\newtoks\@tmp

\def\uplowcase#1{\textsc{#1}}

\newtoks\@last \newtoks\@name

\def\i@uthor#1 #2.mend{\def\paramtw@{#2}\ifx\paramtw@\empty\@last={#1}%
\else\ifx\t@r\empty\@name={#1}\def\t@r{t@r}\else%
\@name=\expandafter{\the\@name{} #1}\fi\i@uthor#2.mend\fi}
\def\inv@uthor#1{\def\t@r{}\@name={}\@last={}\i@uthor#1 .mend%
\expthr@@{\the\@last}, \expthr@@{\the\@name}}

\newcommand{\singlespacing}{\renewcommand{\baselinestretch}{1}%
\let\@tmpcurrentsize=\currentsize \footnotesize \@tmpcurrentsize}
\newcommand{\doublespacing}{\renewcommand{\baselinestretch}{2}
\let\@tmpcurrentsize=\currentsize \footnotesize \@tmpcurrentsize}
\newcommand{\oneandhalfspacing}{\renewcommand{\baselinestretch}{1.15}
\let\@tmpcurrentsize=\currentsize \footnotesize \@tmpcurrentsize}
\def\@prelimsize{\@setsize\@prelimsize{14pt}\xiipt\@xiipt
\let\currentsize=\@prelimsize}

\let\f@@tnote\footnote
\def\footnote#1{\f@@tnote{\singlespacing #1}}

\def\labelenumi{\arabic{enumi}.}
\def\theenumi{\arabic{enumi}}
\def\labelenumii{(\alph{enumii})}
\def\theenumii{\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{\roman{enumiii}.}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{\Alph{enumiv}.}
\def\theenumiv{\Alph{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\bullet$}
\def\labelitemii{\bf --}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

\def\verse{\let\\=\@centercr
 \list{}{\itemsep\z@ \itemindent -1.5em\listparindent \itemindent
 \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em
 \itemindent\listparindent
 \rightmargin\leftmargin\parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\par \singlespacing \list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist

\def\descriptionlabel#1{\hspace\labelsep \bf #1}
\def\description{\list{}{\labelwidth\z@ \itemindent-\leftmargin
 \let\makelabel\descriptionlabel}}
\let\enddescription\endlist
\newdimen\descriptionmargin
\descriptionmargin=3em

\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \else \newpage \fi \@prelimsize \pagestyle{empty}}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi
\pagestyle{plain}\normalsize}

\def\approvalpage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \else \newpage \fi \@prelimsize \pagestyle{empty}}
\def\endapprovalpage{\if@restonecol\twocolumn \else \newpage \fi
\pagestyle{plain}\normalsize}

\arraycolsep 5pt \tabcolsep 6pt \arrayrulewidth .4pt \doublerulesep 2pt
\tabbingsep \labelsep
\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4pt

\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]

\def\thepart {\Roman{part}}
\if@rmnchp \def\thechapter{\Roman{chapter}}
  \else \def\thechapter{\arabic{chapter}} \fi
\def\thesection {\thechapter.\arabic{section}}
\def\thesubsection {\thesection.\arabic{subsection}}
\def\thesubsubsection {\thesubsection .\arabic{subsubsection}}
\def\theparagraph {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph {\theparagraph.\arabic{subparagraph}}
\def\@chapappendix{APPENDIX}

\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{6}

\def\tableofcontents{
\oneandhalfspacing
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\prechapter*{\vskip1.5mm \@tocl \@mkboth{CONTENTS}{CONTENTS}}
\addcontentsline{toc}{leads}{\protect\numberline {}\@tocl}
\ift@rkish
\selectlanguage{turkish}
\fi
 \@starttoc{toc}\if@restonecol\twocolumn\fi\newpage }

\def\l@part#1#2{\addpenalty{-\@highpenalty}
 \addvspace{2.25em plus 1pt} \begingroup
 \@tempdima 3em \parindent \z@ \rightskip \@pnumwidth \parfillskip
-\@pnumwidth
 {\large \bf \leavevmode #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
 \nobreak \endgroup}
\def\l@leads{\vskip 1.0em plus 1pt\@dottedtocline{1}{0em}{0em}}
\def\l@nodots#1#2{\vskip 1.0em plus 1pt\pagebreak[3]\noindent #1\par
  \vskip 0.5em}
\def\l@chapter{\pagebreak[3]
  \vskip 0.5em plus 1pt\@dottedtocline{1}{1.5em}{2.3em}}
\def\l@section{\vskip 0.5em plus 1pt\@dottedtocline{1}{3.8em}{3.5em}}
\def\l@subsection{\vskip 0.5em plus 1pt\@dottedtocline{2}{7.3em}{4.1em}}
\def\l@subsubsection{\vskip 0.5em plus 1pt\@dottedtocline{3}{11.4em}{5em}}
\def\l@paragraph{\vskip 0.5em plus 1pt\@dottedtocline{4}{16.4em}{5em}}
\def\l@subparagraph{\vskip 0.5em plus 1pt\@dottedtocline{5}{21.4em}{6em}}

% \def\appendices{\oneandhalfspacing
% \@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
%  \fi\chapter*{APPENDICES\@mkboth {APPENDICES}{APPENDICES}}
% \addcontentsline{toc}{nodots}{\protect\numberline {}APPENDICES}
%     \setcounter{chapter}{0}\setcounter{section}{0}\setcounter{subsection}{0}
%     \renewcommand{\@chapapp}{\appendixname}
%     \def\thechapter{\Alph{chapter}}
% }
\def\appendix{\par
 \setcounter{chapter}{0}
 \setcounter{section}{0}
 \addcontentsline{toc}{nodots}{\protect\numberline {}APPENDICES}
 \def\@chapapp{APPENDIX}
 \def\thechapter{\Alph{chapter}}}

\def\listoffigures{\oneandhalfspacing
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\prechapter*{\vskip-1mm \@lof\@mkboth
 {\@lof}{\@lof}}
\addcontentsline{toc}{leads}{\protect\numberline {}\@lof}
\@starttoc{lof}\if@restonecol
 \twocolumn\fi\newpage }
\def\l@figure{\@myfiguredottedtocline{1}{0em}{2.0em}}

\def\listoftables{\oneandhalfspacing
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\prechapter*{\vskip-1mm \@lot\@mkboth
 {\@lot}{\@lot}}
\addcontentsline{toc}{leads}{\protect\numberline {}\@lot}
\@starttoc{lot}\if@restonecol
 \twocolumn\fi\newpage }
\def\l@table{\@mytabledottedtocline{1}{0em}{2.0em}}

\def\theglossary#1{\def\he@dgl@{\oneandhalfspacing
\@restonecoltrue \if@twocolumn\@restonecolfalse
 \fi\columnseprule \z@ \columnsep 35pt
 \prechapter*{\vskip 3.5mm \uppercase\expandafter{\@symb@l}%
\@mkboth{\uppercase{\@symb@l}}{\uppercase{\@symb@l}}}%
\parindent \z@ \parskip \z@ plus .3pt\relax }%
 \def\symb@l[##1]{\def\@symb@l{##1}\he@dgl@}%
\@ifnextchar[{\symb@l}{\symb@l[\@loab]}%
\addcontentsline{toc}{leads}{\protect\numberline {}\@loab}
\begin{list}{}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth %labelsep 0.1cm
 \advance\leftmargin\labelsep \itemsep \z@}\def\makelabel##1{##1 \hfil}}%
\def\endtheglossary{\end{list}\if@restonecol\onecolumn\else\clearpage\fi}

%If you have problems with the position of ABBREVATIONS title, you may try to 
%(1) commenting out the above lines, and
%(2) uncomment the following lines.

%\def\glossary{\@ifnextchar[\gl@ssary{\gl@ssary[ABBREVIATIONS]}}
%\def\endglossary{\endgl@ssary}
%\def\gl@ssary[#1]{\@restonecoltrue\if@twocolumn\@restonecolfalse\fi\columnseprule
%\z@
%\columnsep 35pt%\twocolumn
%\chapter*{\vskip *-5mm* LIST OF #1}
%\@mkboth {#1}{#1}
%\addcontentsline{toc}{leads}{\protect\numberline {}LIST OF
%\uppercase{#1}}\singlespacing}
%\def\endgl@ssary{\if@restonecol\onecolumn\else\clearpage\fi}

\def\listofillustrations{\singlespacing
\addcontentsline{toc}{leads}{\protect\numberline {}LIST OF ILLUSTRATIONS}
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn\fi
 \prechapter*{LIST OF ILLUSTRATIONS\@mkboth
 {LIST OF ILLUSTRATIONS}{LIST OF ILLUSTRATIONS}}\@starttoc{loi}\if@restonecol
 \twocolumn\fi\newpage }
\def\l@illustration{\@dottedtocline{1}{1.5em}{2.3em}}


\def\@themybib[#1]#2{\def\@symb@l{#1} \ifx\@symb@l\empty
 \def\@symb@l{REFERENCES} \fi
 \oneandhalfspacing \@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn\fi
 \chapter*{\vskip1mm \uppercase\expandafter{\@symb@l}
 \@mkboth{\uppercase{\@symb@l}}{\uppercase{\@symb@l}}}
  \addcontentsline{toc}{leads}{\protect\numberline {}\uppercase{REFERENCES}}
 \list{[\arabic{enumi}]}{\settowidth\labelwidth{[#2]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep\usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus -.07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\def\thebibliography{
\newpage
\@ifnextchar[{\@themybib}{\@themybib[]}
}
\def\endthebibliography{\endlist}

%\let\endthebibliography=\endlist

\newif\if@restonecol
\def\theindex{\@restonecoltrue\if@twocolumn\@restonecolfalse\fi
\columnseprule \z@
\columnsep 35pt\twocolumn[\@makeschapterhead{Index}]
 \@mkboth{INDEX}{INDEX}\thispagestyle{plain}\parindent\z@
 \parskip\z@ plus .3pt\relax\let\item\@idxitem}
\def\@idxitem{\par\hangindent 40pt}
\def\subitem{\par\hangindent 40pt \hspace*{20pt}}
\def\subsubitem{\par\hangindent 40pt \hspace*{30pt}}
\def\endtheindex{\if@restonecol\onecolumn\else\clearpage\fi}
\def\indexspace{\par \vskip 10pt plus 5pt minus 3pt\relax}

\def\footnoterule{\kern-3\p@
 \hrule width \columnwidth
 \kern 2.6\p@}
\@addtoreset{footnote}{chapter}
\long\def\@makefntext#1{\parindent 1em\noindent
 \hbox to 1.8em{\hss$^{\@thefnmark}$}#1}

\setcounter{topnumber}{2}
\def\topfraction{.7}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.3}
\setcounter{totalnumber}{3}
\def\textfraction{.2}
\def\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}

%captionlar icin buraya bak
\long\def\@makecaption#1#2{
 \ifmy@figur@ \vskip 20pt\fi
 \setbox\@tempboxa\hbox{#1: #2}
 \ifdim \wd\@tempboxa >\hsize #1: #2\par \else \hbox
to\hsize{\hfil\box\@tempboxa\hfil}
 \vskip7.5pt
 \fi \ifmy@t@ble \vskip 20pt\fi}

\newcounter{illustration}[chapter]
\def\theillustration{\thechapter.\@arabic\c@illustration}
\def\fps@illustration{tbp}
\def\ftype@illustration{3}
\def\ext@illustration{loi}
\def\fnum@illustration{Illustration \theillustration}
\def\illustration{\singlespacing \my@figur@true \@float{illustration}}
\let\endillustration\end@float
\@namedef{illustration*}{\@dblfloat{illustration}}
\@namedef{endillustration*}{\end@dblfloat}

\newcounter{figure}[chapter]
\def\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{Figure \thefigure}
\def\figure{\oneandhalfspacing \my@figur@true \@float{figure}}
\let\endfigure\end@float
\@namedef{figure*}{\@float{figure}}
\@namedef{endfigure*}{\end@float}

\newcounter{table}[chapter]
\def\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\@tablelabel\space\thetable}
\def\table{\oneandhalfspacing \my@t@bletrue \@float{table}}
\let\endtable\end@float
\@namedef{table*}{\@float{table}}
\@namedef{endtable*}{\end@float}


\def\@graduateschool{Applied Mathematics}
%\def\@graduateschool{Natural and Applied Sciences}
\def\@tgraduateschool{Fen Bilimleri Enstitüsü}
\ifp@hd
\def\@shortdegree{Ph.D.}
\def\@degree{Doctor of Philosophy}
\def\@tdegree{Doktora}
\else
\def\@shortdegree{M.S.}
\def\@degree{Master of Science}
\def\@tdegree{Yüksek Lisans}
\fi

\def\department#1{\def\@department{#1}}
\def\@departmentofsupervisor{}\def\departmentofsupervisor#1{\def\@departmentofsupervisor{#1}}
\def\turkishdepartmentofsupervisor#1{\def\@turkishdepartmentofsupervisor{#1}}
\def\@departmentofcosupervisor{}\def\departmentofcosupervisor#1{\def\@departmentofcosupervisor{#1}}
\def\turkishdepartmentofcosupervisor#1{\def\@turkishdepartmentofcosupervisor{#1}}
\def\turkishdepartment#1{\def\@tdepartment{#1}}
\def\@ttitle{}\def\turkishtitle#1{\def\@ttitle{#1}}

\def\@proftitlename{Prof. Dr.}\def\@tproftitlename{Prof. Dr.}
\def\@assocproftitlename{Assoc. Prof. Dr.}\def\@tassocproftitlename{Doç. Dr.}
\def\@assistproftitlename{Assist. Prof. Dr.}\def\@tassistproftitlename{Yrd. Doç. Dr.}
\def\@drtitlename{Dr.}\def\@tdrtitlename{Dr.}

\def\@director{} 
\def\director{\@ifnextchar[{\@academicdirector}{\@plaindirector}}
\def\@plaindirector#1{\def\@director{#1}}
\def\@academicdirector[#1]#2{\def\@director{\csname @#1titlename\endcsname\space#2}}

\def\@headofdept{} 
\def\headofdept{\@ifnextchar[{\@academicheadofdept}{\@plainheadofdept}}
\def\@plainheadofdept#1{\def\@headofdept{#1}}
\def\@academicheadofdept[#1]#2{\def\@headofdept{\csname @#1titlename\endcsname\space#2}}

\def\@supervisor{}\def\@turkishsupervisor{} 
\def\supervisor{\@ifnextchar[{\@academicsupervisor}{\@plainsupervisor}}
\def\@plainsupervisor#1{\def\@supervisor{#1} \def\@turkishsupervisor{#1}}
\def\@academicsupervisor[#1]#2{
	\def\@supervisor{\csname @#1titlename\endcsname\space#2}
	\def\@turkishsupervisor{\csname @t#1titlename\endcsname\space#2}
}
\def\turkishsupervisor#1{\def\@turkishsupervisor{#1}}

\newif\ifexist@co \exist@cofalse
\newif\ifcommitteeiv \committeeivtrue
%\newif\ifexist@civ \exist@civtrue
\def\@cosupervisor{} \def\@turkishcosupervisor{} 
\def\cosupervisor{\@ifnextchar[{\@academiccosupervisor}{\@plaincosupervisor}}
\def\@plaincosupervisor#1{\def\@cosupervisor{#1} \def\@turkishcosupervisor{#1}\exist@cotrue }
\def\@academiccosupervisor[#1]#2{
	\def\@cosupervisor{\csname @#1titlename\endcsname\space#2}
	\def\@turkishcosupervisor{\csname @t#1titlename\endcsname\space#2}\exist@cotrue 
}
\def\turkishcosupervisor#1{\def\@turkishcosupervisor{#1}}

\def\@affiliationi{} \def\affiliationi#1{\def\@affiliationi{#1}}
\def\@affiliationii{} \def\affiliationii#1{\def\@affiliationii{#1}}
\def\@affiliationiii{} \def\affiliationiii#1{\def\@affiliationiii{#1}}
\def\@affiliationiv{} \def\affiliationiv#1{\def\@affiliationiv{#1}}
\def\@affiliationv{} \def\affiliationv#1{\def\@affiliationv{#1}}
\def\@taffiliationi{} \def\turkishaffiliationi#1{\def\@taffiliationi{#1}}
\def\@taffiliationii{} \def\turkishaffiliationii#1{\def\@taffiliationii{#1}}
\def\@taffiliationiii{} \def\turkishaffiliationiii#1{\def\@taffiliationiii{#1}}
\def\@taffiliationiv{} \def\turkishaffiliationiv#1{\def\@taffiliationiv{#1}}
\def\@taffiliationv{} \def\turkishaffiliationv#1{\def\@taffiliationv{#1}}

\def\@committeememberi{} \def\@tcommitteememberi{} 
\def\@committeememberii{} \def\@tcommitteememberii{} 
\def\@committeememberiii{} \def\@tcommitteememberiii{} 
\def\@committeememberiv{} \def\@tcommitteememberiv{} 
\def\@committeememberv{} \def\@tcommitteememberv{} 
\def\committeememberi{\@ifnextchar[{\@academiccommitteemember{i}}{\@plaincommitteemember{i}}}
\def\committeememberii{\@ifnextchar[{\@academiccommitteemember{ii}}{\@plaincommitteemember{ii}}}
\def\committeememberiii{\@ifnextchar[{\@academiccommitteemember{iii}}{\@plaincommitteemember{iii}}}

%\def\committeememberiv{\@ifnextchar[{\@academiccommitteemember{iv}}{\@plaincommitteemember{iv}}\exist@civtrue }
%\def\@committeememberiv#1{\def\@committeememberiv{#1} \def\@tcommitteememberiv{#1}\exist@civtrue }


\def\committeememberiv{\@ifnextchar[{\@academiccommitteemember{iv}}{\@plaincommitteemember{iv}}}
\def\committeememberv{\@ifnextchar[{\@academiccommitteemember{v}}{\@plaincommitteemember{v}}}
\def\@plaincommitteemember#1#2{\expandafter\def\csname @committeemember#1\endcsname{#2} \expandafter\def\csname @tcommitteemember#1\endcsname{#2}}
\def\@academiccommitteemember#1[#2]#3{
	\expandafter\def\csname @committeemember#1\endcsname{\csname @#2titlename\endcsname\space#3}
	\expandafter\def\csname @tcommitteemember#1\endcsname{\csname @t#2titlename\endcsname\space#3}
}
\def\turkishcommitteememberi#1{\def\@tcommitteememberi{#1}}
\def\turkishcommitteememberii#1{\def\@tcommitteememberii{#1}}
\def\turkishcommitteememberiii#1{\def\@tcommitteememberiii{#1}}
\def\turkishcommitteememberiv#1{\def\@tcommitteememberiv{#1}}
\def\turkishcommitteememberv#1{\def\@tcommitteememberv{#1}}

\def\@keywords{} 
\def\keywords#1{\def\@keywords{#1}}
\def\@anahtarklm{} \def\anahtarklm#1{\def\@anahtarklm{#1}}

\def\maketitle{\begin{titlepage}\normalsize
\let\footnotesize\small \let\footnoterule\relax \setcounter{page}{1}
\null \nointerlineskip
\vskip 0.8cm \begin{center} \nointerlineskip
 \vbox{\singlespacing \MakeUppercase{\@disptitle}}
\vskip 4.5cm plus0pt minus1cm \nointerlineskip
 \vbox{\singlespacing \@titleparta}
\vskip 1.85cm plus0pt minus2mm \nointerlineskip
 \vbox{\@titleseperator}
\vskip 1.85cm plus0pt minus2mm \nointerlineskip
 \vbox{\expandafter{\MakeUppercase{\@author}}}
\vskip 5cm plus0pt minus1cm \nointerlineskip
 \vbox{\singlespacing \@titlepartb}
\vskip 5cm plus0pt minus1cm \nointerlineskip
\vspace{\stretch{1}}
 \vbox{
\ifx\kernel@date\@date
	\ift@rkish\MakeUppercase{\@tdate}\else\MakeUppercase{\@date}\fi
	\else
	\ift@rkish\@datetoturuc{\@date}\else\MakeUppercase{\@date}\fi
	\fi
} \end{center}
\end{titlepage}
\setcounter{footnote}{0} \let\thanks\relax
\gdef\@thanks{}\let\maketitle\relax}

\def\makeapproval{\begin{approvalpage} \normalsize \parskip \z@
\parskip \z@ \topsep \z@ \partopsep \z@ \parsep \z@ \itemsep \z@
\null \nointerlineskip \parindent=0pt
\vskip1.0cm plus0pt minus2mm \nointerlineskip
\parbox[t]{15cm}{\begin{center} \@approvallabel\end{center}}
\vskip1.0cm plus0pt minus2mm \nointerlineskip
\parbox[t]{15cm}{\begin{center}\singlespacing \bf{\MakeUppercase{\@disptitle}}\end{center}}
\vskip2.0cm plus0pt minus2mm \nointerlineskip
\parbox[t]{14.5cm}{\singlespacing \@approvaltext}
\vskip1.5cm plus0pt minus2mm \nointerlineskip
\vbox{\parbox[t]{11cm}{\singlespacing \expthr@@{\@director}\\
      \@directoraff} \hfill
      \parbox[t]{3cm}{\singlespacing \begin{center}
      \rule{3cm}{1pt} \\
      \end{center}\nointerlineskip}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
 \vbox{\parbox{11cm}{\singlespacing \expthr@@{\@headofdept}\\
	\@chairlabel, {\bf \expthr@@{\@dispdepartment}}} \hfill \parbox{3cm}{\singlespacing \begin{center}
                      \rule{3cm}{1pt}
                      \end{center}\nointerlineskip}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
\vbox{
		\parbox{11cm}{\singlespacing \expthr@@{\@dispsupervisor}\\
		\@supervisorlabel, {\bf \expthr@@{\@dispdepartmentofsupervisor}}} \hfill \parbox{3cm}{\singlespacing \begin{center}
                      \rule{3cm}{1pt}
                      \end{center}\nointerlineskip}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
\ifexist@co
  \vbox{
  \parbox{11cm}{\singlespacing \expthr@@{\@cosupervisor}\\ \@cosupervisorlabel, {\bf
\expthr@@{\@dispdepartmentofcosupervisor}}} \hfill \parbox{3cm}{\singlespacing \begin{center} \rule{3cm}{1pt} \end{center}\nointerlineskip}}
\else
  \vbox{\parbox{11cm}{~\\~} \hfill \parbox{3cm}{~\\~}}
\fi
\ifcommitteeiv\else\vskip2.7cm plus0pt minus2mm \nointerlineskip\fi
\vskip1.5cm plus0pt minus2mm \nointerlineskip
 \vbox{\bf{\@committeelabel:}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
 \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberi \\ \expthr@@{\@dispaffiliationi}}
 \hfill \rule{3cm}{1pt}}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
 \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberii \\ \expthr@@{\@dispaffiliationii}}
 \hfill \rule{3cm}{1pt}}}
\vskip0.5cm plus0pt minus2mm \nointerlineskip
\ifcommitteeiv
 \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberiii \\ \expthr@@{\@dispaffiliationiii}}
 \hfill \rule{3cm}{1pt}}}
  \vskip0.5cm plus0pt minus2mm \nointerlineskip
   \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberiv \\ \expthr@@{\@dispaffiliationiv}}
  \hfill \rule{3cm}{1pt}}
  \vskip0.5cm plus0pt minus2mm \nointerlineskip
   \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberv \\ \expthr@@{\@dispaffiliationv}}
  \hfill \rule{3cm}{1pt}}}}
\else
 \vbox{\parbox{14.25cm}{\singlespacing \expthr@@{\@dispcommitteememberiii \\ \expthr@@{\@dispaffiliationiii}}
 \hfill \rule{3cm}{1pt}}}
\fi
\vskip1.5cm plus0pt minus2mm \nointerlineskip
 \vbox{\parbox{14.25cm}{\bf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\@datelabel:
 \hfill \rule{3cm}{1pt}}}
\end{approvalpage}
\gdef\@director{}\let\director\relax
\gdef\@headofdept{}\let\headofdept\relax
\gdef\@degree{}
\gdef\@committeemember1{}\let\committeemember1\relax
\gdef\@committeemember2{}\let\committeemember2\relax
\gdef\@committeemember3{}\let\committeemember3\relax
\gdef\@committeemember4{}\let\committeemember4\relax
\gdef\@committeemember5{}\let\committeemember5\relax
\gdef\@graduateschool{}\let\graduateschool\relax \let\makeapproval\relax}

\def\turmonthjanuary{Ocak}
\def\turmonthfebruary{Şubat}
\def\turmonthmarch{Mart}
\def\turmonthapril{Nisan}
\def\turmonthmay{Mayıs}
\def\turmonthjune{Haziran}
\def\turmonthjuly{Temmuz}
\def\turmonthaugust{Ağustos}
\def\turmonthseptember{Eylül}
\def\turmonthoctober{Ekim}
\def\turmonthnovember{Kasım}
\def\turmonthdecember{Aralık}
\def\@trmonth#1{\csname turmonth#1\endcsname}
\def\@helperdatetoturuc#1 #2\nil{
\lowercase{\def\@temp{#1}}
\MakeUppercase{\@trmonth{\@temp}}~#2}
\def\@helperdatetotur#1 #2\nil{
\lowercase{\def\@temp{#1}}
\@trmonth{\@temp}~#2}
\def\@datetotur#1{\expandafter\@helperdatetotur#1\nil}
\def\@datetoturuc#1{\expandafter\@helperdatetoturuc#1\nil}

\def\showoz{%\addcontentsline{toc}{leads}{\protect\numberline {}\"OZ}
\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
 \fi
 \prechapter*{\vskip2.5mm \"{O}Z}
\selectlanguage{turkish}
\@mkboth {\"{O}Z}{\"{O}Z} \vskip 1cm
\addcontentsline{toc}{leads}{\protect\numberline {}\"OZ}
\begin{center}\singlespacing \MakeUppercase{\expandafter{\@ttitle}} \\ \vskip 1.0\baselineskip \vskip 1.5cm
 \expandafter\expandafter\expandafter\inv@uthor\expandafter{\@author} \\[0.2cm]
 \begin{tabular}{ll}
 \multicolumn{2}{l}{\@tdegree, {\@tdepartment} Bölümü}\\[0.2cm]
 Tez Y\"{o}neticisi &: \expthr@@{\@turkishsupervisor} \\[0.2cm]
 \ifexist@co Ortak Tez Y\"{o}neticisi &: \expthr@@{\@turkishcosupervisor}\fi \\[0.2cm]
\end{tabular}
\vskip \baselineskip \ifx\@date\kernel@date \@tdate \else \@datetotur{\@date}\fi, \pageref{`@lastpage'} sayfa\end{center}
\vskip \baselineskip \oneandhalfspacing
\ifd@uble\doublespacing\else\ifs@ngle\singlespacing\else\oneandhalfspacing\fi\fi
\@oztext}
\def\endshowoz{\vskip 1.5\baselineskip \par \noindent Anahtar Kelimeler: \@anahtarklm
\gdef\@anahtarklm{}\let\anahtarklm\relax
\gdef\@tdegree{}
\gdef\@tdepartment{}\let\turkishdepartment\relax
\gdef\@ttitle{}\let\turkishtitle\relax
\gdef\@supervisor{}\let\supervisor\relax
\gdef\@cosupervisor{}\let\cosupervisor\relax
\if@restonecol \twocolumn\fi\oneandhalfspacing\newpage
\ift@rkish\relax\else\selectlanguage{english}\fi }

\newcommand{\abstract}[1]{\newcommand{\@abstracttext}{#1}}
\newcommand{\oz}[1]{\newcommand{\@oztext}{#1}}
\newcommand{\dedication}[1]{\newcommand{\@dedicationtext}{#1}}
\newcommand{\turkishdedication}[1]{\newcommand{\@tdedicationtext}{#1}}
\newcommand{\turkishacknowledgments}[1]{\newcommand{\@tacknowledgmentstext}{#1}}
\newcommand{\acknowledgments}[1]{\newcommand{\@acknowledgmentstext}{#1}}

\def\showabstract{
\setlength{\parindent}{0em}
\setlength{\parskip}{10pt}
%\addcontentsline{toc}{leads}{\protect\numberline {}ABSTRACT}
\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
 \fi
 \prechapter*{\vskip4mm ABSTRACT}\@mkboth {ABSTRACT}{ABSTRACT} \vskip 1cm
%omur
\addcontentsline{toc}{leads}{\protect\numberline {}ABSTRACT}

\selectlanguage{english}
 \begin{center}\uppercase\expandafter{\@title} \\ \vskip 1.0\baselineskip \vskip 1.5cm
 \expandafter\expandafter\expandafter\inv@uthor\expandafter{\@author} \\[0.2cm]
 \begin{tabular}{ll}
 \multicolumn{2}{l}{\@shortdegree, Department of \@department} \\[0.2cm]
 Supervisor &: \expthr@@{\@supervisor} \\[0.2cm]
 \ifexist@co Co-Supervisor &: \expthr@@{\@cosupervisor}\fi\\[0.2cm]
\end{tabular}
\vskip \baselineskip \@date, \pageref{`@lastpage'} pages\end{center}
\vskip \baselineskip 
\ifd@uble\doublespacing\else\ifs@ngle\singlespacing\else\oneandhalfspacing\fi\fi
\@abstracttext}

\def\endshowabstract{
\vskip 1.5\baselineskip \par \noindent Keywords: \@keywords
\gdef\@keywords{}\let\keywords\relax
\gdef\@shortdegree{}
\gdef\@department{}\let\department\relax
\gdef\@supervisor{}\let\supervisor\relax
\gdef\@cosupervisor{}\let\cosupervisor\relax
\if@restonecol \twocolumn\fi\newpage 
\ift@rkish\selectlanguage{turkish}\fi}

\def\showdedication{
\begin{center}\null
\vspace*{\fill}
\ift@rkish
\@tdedicationtext
\else
\@dedicationtext
\fi}

\def\endshowdedication{
\vspace*{\fill}
\end{center}
\newpage}

\def\plagiarism{
\null
\vspace*{\fill}
\vspace*{\fill}
\vspace*{\fill}
\vspace*{\fill}
\vspace*{\fill}
\parindent0cm
%\ifd@uble\doublespacing\else\ifs@ngle\singlespacing\else\oneandhalfspacing\fi\fi
\singlespacing
{\bf \@plagirismtext}\\[1.5cm]
\begin{flushleft}
\hfill{
\begin{tabular}{l@{}ll}
\textnormal{\@plagirismnametext} & \textnormal{:} & \textnormal\bf{\MakeUppercase{\@author}} \\[1cm]
\textnormal{\@plagirismsignaturetext} & \textnormal{:} &
\end{tabular}
}
\vspace*{\fill}
\end{flushleft}
\newpage}

\newcommand{\@showturkishacknowledgments}{
\setlength{\parindent}{0em}
\setlength{\parskip}{10pt}

\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
\fi
\vskip6.5mm
\prechapter*{TEŞEKKÜRLER} \selectlanguage{turkish}\@mkboth{TEŞEKKÜRLER}{TEŞEKKÜRLER}
\addcontentsline{toc}{leads}{\protect\numberline{}TEŞEKKÜRLER}
\ifd@uble\doublespacing\else\ifs@ngle\singlespacing\else\oneandhalfspacing\fi\fi
\@tacknowledgmentstext
\if@restonecol
    \twocolumn
\fi
\ift@rkish
\relax
\else
\selectlanguage{english}
\fi
\newpage
\setlength{\parindent}{0em}
\setlength{\parskip}{3pt}
}

\newcommand{\@showacknowledgments}{
\setlength{\parindent}{0em}
\setlength{\parskip}{10pt}
\ifd@uble\doublespacing\else\ifs@ngle\singlespacing\else\oneandhalfspacing\fi\fi

\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
\fi
\vskip6.5mm
\prechapter*{ACKNOWLEDGMENTS} 
\@mkboth{ACKNOWLEDGMENTS}{ACKNOWLEDGMENTS}
\addcontentsline{toc}{leads}{\protect\numberline{}ACKNOWLEDGMENTS}
\oneandhalfspacing
\@acknowledgmentstext
\if@restonecol
    \twocolumn
\fi
\newpage
\setlength{\parindent}{0em}
\setlength{\parskip}{3pt}
}

%\def\acknowledgments{
%\addcontentsline{toc}{leads}{\protect\numberline {}ACKNOWLEDGMENTS}
%\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
%\fi\prechapter*{ACKNOWLEDGMENTS}\@mkboth {ACKNOWLEDGMENTS}{ACKNOWLEDGMENTS}}
%\def\endacknowledgments{\if@restonecol \twocolumn\fi\doublespacing\newpage}

\newenvironment{preface}{
\addcontentsline{toc}{leads}{\protect\numberline {}PREFACE}
\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
 \fi
\prechapter*{PREFACE}\@mkboth {PREFACE}{PREFACE}
}{
\if@restonecol
    \twocolumn
\fi
\newpage
}

\newenvironment{curriculumvitae}{
\@restonecolfalse
\if@twocolumn
    \@restonecoltrue\onecolumn
\fi
\chapter*{CURRICULUM VITAE} \@mkboth{CURRICULUM VITAE}{CURRICULUM VITAE}
\addcontentsline{toc}{leads}{\protect\numberline{}CURRICULUM VITAE}
}{
\if@restonecol
    \twocolumn
\fi

}

\def\vita{\addcontentsline{toc}{leads}{\protect\numberline {}VITA}
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \pagestyle{plain}
 \fi\chapter*{VITA}\@mkboth {VITA}{VITA}}
\def\endvita{\if@restonecol \twocolumn\fi}

\def\preliminaries{\pagenumbering{roman}\pagestyle{plain}
\ift@rkish
\selectlanguage{turkish}
\def\@disptitle{\@ttitle}
\def\@dispdepartment{\@tdepartment}
\def\@dispsupervisor{\@turkishsupervisor}
\def\@dispdepartmentofsupervisor{\@ifundefined{@turkishdepartmentofsupervisor}{\@departmentofsupervisor}{\@turkishdepartmentofsupervisor}}
\def\@dispdepartmentofcosupervisor{\@ifundefined{@turkishdepartmentofcosupervisor}{\@departmentofcosupervisor}{\@turkishdepartmentofcosupervisor}}
\def\@dispaffiliationi{\@taffiliationi}
\def\@dispaffiliationii{\@taffiliationii}
\def\@dispaffiliationiii{\@taffiliationiii}
\def\@dispaffiliationiv{\@taffiliationiv}
\def\@dispaffiliationv{\@taffiliationv}
\def\@dispcommitteememberi{\ifx\@tcommitteememberi\empty\@committeememberi\else\@tcommitteememberi\fi}
\def\@dispcommitteememberii{\ifx\@tcommitteememberii\empty\@committeememberii\else\@tcommitteememberii\fi}
\def\@dispcommitteememberiii{\ifx\@tcommitteememberiii\empty\@committeememberiii\else\@tcommitteememberiii\fi}
\def\@dispcommitteememberiv{\ifx\@tcommitteememberiv\empty\@committeememberiv\else\@tcommitteememberiv\fi}
\def\@dispcommitteememberv{\ifx\@tcommitteememberv\empty\@committeememberv\else\@tcommitteememberv\fi}
\else
\selectlanguage{english}
\def\@disptitle{\@title}
\def\@dispdepartment{\@department}
\def\@dispsupervisor{\@supervisor}
\def\@dispdepartmentofsupervisor{\@departmentofsupervisor}
\def\@dispdepartmentofcosupervisor{\@departmentofcosupervisor}
\def\@dispaffiliationi{\@affiliationi}
\def\@dispaffiliationii{\@affiliationii}
\def\@dispaffiliationiii{\@affiliationiii}
\def\@dispaffiliationiv{\@affiliationiv}
\def\@dispaffiliationv{\@affiliationv}
\def\@dispcommitteememberi{\@committeememberi}
\def\@dispcommitteememberii{\@committeememberii}
\def\@dispcommitteememberiii{\@committeememberiii}
\def\@dispcommitteememberiv{\@committeememberiv}
\def\@dispcommitteememberv{\@committeememberv}
\fi
\maketitle
\clearpage
\thispagestyle{empty}
\mbox{}
\makeapproval
\cleardoublepage
\plagiarism
\cleardoublepage
\showabstract
\endshowabstract
\cleardoublepage
\showoz
\endshowoz
\showdedication
\endshowdedication
\cleardoublepage
\ift@rkish
\@showturkishacknowledgments
\else
\@showacknowledgments
\fi
\cleardoublepage
\tableofcontents
\cleardoublepage
\listoftables
\listoffigures
\if@lg@rithmloaded
\renewcommand{\listalgorithmcfname}{LIST OF ALGORITHMS}
\makeatletter
\def\listofalgorithms{\oneandhalfspacing
\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi
 \ifturkish
 \prechapter*{\vskip-1mm ALGORİTMALAR LİSTESİ\@mkboth
 {ALGORİTMALAR LİSTESİ}{ALGORİTMALAR LİSTESİ}}
\addcontentsline{toc}{leads}{\protect\numberline {}ALGORİTMALAR LİSTESİ}
\else
 \prechapter*{\vskip-1mm LIST OF ALGORITHMS\@mkboth
 {LIST OF ALGORITHMS}{LIST OF ALGORITHMS}}
\addcontentsline{toc}{leads}{\protect\numberline {}LIST OF ALGORITHMS}
\fi
\@starttoc{loa}\if@restonecol
 \twocolumn\fi\newpage 
 \ifturkish
 \addtocontents{loa}{\protect\numberline{}ALGORİTMALAR}
 \addtocontents{loa}{\par}
%\addcontentsline{loa}{nodots}{\protect\numberline{}ALGORİTMALAR}
\else
%\addcontentsline{loa}{nodots}{\protect\numberline{}ALGORITHMS}
%\addtocontents{loa}{\protect\numberline{}ALGORITHMS}
%\addtocontents{loa}{\par}
\fi}
\def\l@algocf{\@myalgodottedtocline{1}{0em}{2.0em}}
\makeatother
\listofalgorithms
\fi
}%
\def\endpreliminaries{
\if@twoside\cleardoublepage\else\newpage\fi\pagestyle{headings}\pagenumbering{arabic}%
\addtocontents{toc}{\protect\numberline {}\@chapterslabel}
%\addtocontents{lof}{\protect\numberline {}\@figureslabel}
%\addtocontents{lof}{\par}
%\addtocontents{lot}{\protect\numberline {}\@tableslabel}
%\addtocontents{lot}{\par}
\ift@rkish
\shorthandoff{=}
\shorthandoff{:}
\shorthandoff{!}
\fi
}

\mark{{}{}}

\def\ps@headings{\let\@mkboth\markboth
\def\@oddfoot{}\def\@evenfoot{}%
\ifpn@topr \def\@evenhead{\hbox{} \hfil \rm\thepage}
\else \def\@evenhead{\hbox{} \hfil \rm\thepage \hfil} \fi
\let\@oddhead=\@evenhead%
\def\chaptermark##1{\markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
 \@chapapp\ \thechapter. \ \fi ##1}}{}}%
\def\sectionmark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\z@
 \thesection. \ \fi ##1}}}}

\def\ps@plain{\let\@mkboth\markboth
\def\@evenfoot{\hbox{} \hfil \rm\thepage \hfil} \let\@oddfoot=\@evenfoot%
\def\@evenhead{} \let\@oddhead=\@evenhead%
\def\chaptermark##1{\markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
 \@chapapp\ \thechapter. \ \fi ##1}}{}}%
\def\sectionmark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\z@
 \thesection. \ \fi ##1}}}}

\def\ps@myheadings{\let\@mkboth\@gobbletwo
\def\@oddhead{\hbox{}\sl\rightmark \hfil
\rm\thepage}\def\@oddfoot{}\def\@evenhead{\rm \thepage\hfil\sl\leftmark\hbox
{}}\def\@evenfoot{}\def\sectionmark##1{}\def\subsectionmark##1{}}

\def\today{\ifcase\month\or
 January\or February\or March\or April\or May\or June\or
 July\or August\or September\or October\or November\or December\fi~\number\year}

\def\bugun{\ifcase\month\or
 Ocak\or \c{S}ubat\or Mart\or Nisan\or May{\i}s\or Haziran\or
 Temmuz\or A\u{g}ustos\or Eyl\"{u}l\or Ekim\or Kas{\i}m\or Aral{\i}k\fi
 \space\number\year}

\let\@date\today
\let\kernel@date\today
\let\@tdate\bugun

\let\orighead\@makechapterhead
\renewcommand*{\@makechapterhead}[1]{\orighead{\MakeUppercase{#1}}}


\@addtoreset{equation}{chapter}
\def\theequation{\thechapter.\arabic{equation}}

\ifpn@top \pagestyle{headings} \else \pagestyle{plain} \fi
\pagenumbering{arabic} \onecolumn

\textheight=23.5cm

\uccode`@=`I \lccode`@="10

\let\@enddoc=\enddocument
\def\enddocument{\label{`@lastpage'}\@enddoc }

