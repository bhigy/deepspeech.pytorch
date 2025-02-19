PHONEMES="""arpabet	ipa	class
aa	ɑ	vowel
ae	æ	vowel
ah	ə	vowel
ao	ɔ	vowel
aw	aʊ	vowel
ay	aɪ	vowel
b	b	plosive
ch	tʃ	affricate
d	d	plosive
dh	ð	fricative
eh	ɛ	vowel
er	ɚ	vowel
ey	e	vowel
f	f	fricative
g	g	plosive
hh	h	fricative
ih	ɪ	vowel
iy	i	vowel
jh	dʒ	affricate
k	k	plosive
l	l	approximant
m	m	nasal
n	n	nasal
ng	ŋ	nasal
ow	o	vowel
oy	ɔɪ	vowel
p	p	plosive
r	ɹ	approximant
s	s	fricative
sh	ʃ	fricative
t	t	plosive
th	θ	fricative
uh	ʊ	vowel
uw	u	vowel
v	v	fricative
w	w	approximant
y	j	approximant
z	z	fricative
zh	ʒ	fricative"""

def parseipa():
    mapping =  {}
    lines = PHONEMES.split("\n")
    for line in lines[1:]:
        arpa, ipa, _ = line.split()
        mapping[arpa] = ipa
    return mapping

_arpa2ipa = parseipa()

def arpa2ipa(arpa, default=None):
    return _arpa2ipa.get(arpa, default) 

