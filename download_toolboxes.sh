#!/bin/sh

curl -sS http://www.jonathanleroux.org/software/sparseNMF.zip > sparseNMF.zip
unzip sparseNMF.zip -d sparseNMF/
rm sparseNMF.zip

mkdir evaluation

mkdir evaluation/bss_eval
curl -sS http://bass-db.gforge.inria.fr/bss_eval/bss_eval.zip > bss_eval.zip
unzip bss_eval.zip -d evaluation/bss_eval/
rm bss_eval.zip

mkdir evaluation/composite
curl -sS http://ecs.utdallas.edu/loizou/speech/composite.zip > composite.zip
unzip composite.zip -d evaluation/composite/
rm composite.zip

mkdir evaluation/stoi
curl -sS http://ceestaal.nl/stoi.zip > stoi.zip
unzip stoi.zip -d evaluation/stoi/
rm stoi.zip

mkdir evaluation/voicebox
curl -sS http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.zip > voicebox.zip
unzip voicebox.zip -d evaluation/voicebox/
rm voicebox.zip

