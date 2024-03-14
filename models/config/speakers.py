import json
from typing import Dict, List

# Load the ID mapping
with open("training/datasets/speaker_id_mapping_libri.json") as f:
    id_mapping = json.load(f)

# Create a reverse mapping
reverse_mapping: Dict[int, int] = {int(v): int(k) for k, v in id_mapping.items()}

# Selected for the fine-tuning
# train-960 subset of LibriTTS
selected_speakers = [
    574, #	Daniel Shorten 	M 	train-clean-100
    242, #	J. Hall 	M 	train-other-500
    536, #	Robert Flach 	M 	train-other-500
    82,  #	Andy Minter 	M 	train-other-500
    672, #	Stuart Bell 	M 	train-other-500
    315, #	Jean Crevier 	M 	train-other-500
    628, #	Bryan Ness 	M 	train-clean-100
    61,  #	John Greenman 	M 	train-other-500
    649, #	Scarlett! 	F 	train-clean-360
    105, #	Marian Brown 	F 	train-clean-360
    399, #	entada 	F 	train-clean-360
    89,  # 	Paula Berinstein 	F 	train-clean-360
    502, #	Lee Elliott 	F 	train-other-500
    102, #	Maureen S. O'Brien 	F 	train-clean-100
    544, #	Miranda Stinson 	F 	train-clean-360
    653, #	cucciasv 	F 	train-other-500
    465, #	Leonie Rose 	F 	train-clean-100
    96,  #	Kymm Zuckert 	F 	train-other-500
    447, #	Lee Ann Howlett 	F 	train-clean-360
    165, # 	Elisabeth Shields 	F 	train-clean-100
    430, # 	Millbeach 	F 	train-other-500
    214, #	Scott Splavec 	M 	train-clean-100
    666, #	Kelly Dougherty 	M 	train-clean-360
    481, #	Scott Sherris 	M 	train-clean-360
    463, #	Chris Hughes 	M 	train-other-500
    273, #	Andrew Lebrun 	M 	train-other-500
    172, #	Harvey Chinn 	M 	train-other-500
    83,  #	Graham Williams 	M 	train-other-500
    523, #	Michael Loftus 	M 	train-clean-360
    38,  #	Kurt Copeland 	M 	train-clean-360
    248, #	fieldsofgold 	M 	train-other-500
    234, #	Menno 	M 	train-other-500
    145, #	Mr. Baby Man 	M 	train-clean-360
    250, #	Quentin 	M 	train-clean-360
    498, #	Chris Gladis 	M 	train-clean-100
    123, #	Sean McGaughey 	M 	train-clean-360
    171, #	Paul Harvey 	M 	train-clean-360
    49,  # 	Kristen McQuillin 	F 	train-clean-100
    588, # 	Kalynda 	F 	train-clean-360
    117, # 	Caitlin Kelly 	F 	train-clean-360
    657, #	Shannon 	F 	train-other-500
    275, #	Zale Schafer (Rose May Chamberlin Memorial Foundat 	F 	train-clean-360
    604, #	Anne-Marie 	F 	train-other-500
    64,  #	Christiane Levesque 	F 	train-clean-360
    685, #	Nikki Sullivan 	F 	train-clean-100
    355, #	Lana Taylor 	F 	train-clean-100
    185, #	Kim Braun 	F 	train-clean-360
    52,  #	Cori Samuel 	F 	train-other-500
    218, #	Joy Chan 	F 	train-other-500
    549, #	AmyAG 	F 	train-other-500
    617, #	PJ 	F 	train-other-500
    414, #	Christabel 	F 	train-clean-100
    382, #	Kelli Robinson 	F 	train-clean-360
    76,  # 	ML Cohen 	M 	train-other-500
    176, #	Micah Sheppard 	M 	train-clean-360
    233, #	mikenkat 	M 	train-clean-360
    390, #	JimmyLogan 	M 	train-clean-360
    393, # 	Tim Lundeen 	M 	train-clean-360
    425, #	RedToby 	M 	train-clean-360
    398, #	Sam Fold 	M 	train-other-500
    372, #	Jim Mullins 	M 	train-clean-360
    99,  #	Stewart Wills 	M 	train-clean-100
    340, # 	Nick Gallant 	M 	train-clean-100
    40,  #	JemmaBlythe 	F 	train-other-500
    118, # 	Brenda Dayne 	F 	train-clean-360
    640, #	David A. Stokely 	M 	train-other-500
    50,  #	Dan Threetrees 	M 	train-clean-360
    373, #	Brooks Seveer 	M 	train-clean-360
    124, #	Steve Karafit 	M 	train-clean-100
    314, #	Carl Vonnoh, III 	M 	train-clean-360
    531, #	Fr. Richard Zeile of Detroit 	M 	train-other-500
    383, #	Mike Roop 	M 	train-other-500
    710, #	Sheila Morton 	F 	train-clean-100
    450, #	Heather Duncan 	F 	train-clean-360
    645, #	Micah 	M 	train-other-500
    517, #	Madame Tusk 	F 	train-other-500
    479, #	Wina Hathaway 	F 	train-other-500
    30,  #	Ophelia Darcy 	F 	train-other-500
    220, #	Tina Tilney 	F 	train-clean-360
    63,  #	Linda Wilcox 	F 	train-other-500
    283, #	Bethany Simpson 	F 	train-clean-360
    644, #	Cynthia Zocca 	F 	train-clean-360
    677, #	Allyson Hester 	F 	train-other-500
    21,  #	Kelly Bescherer 	F 	train-other-500
    552, #	Mim Ritty 	F 	train-clean-100
    80,  # 	Fox in the Stars 	F 	train-clean-100
    394, #	swroot 	F 	train-clean-360
    426, #	Megan Stemm-Wade 	F 	train-clean-100
    91,  #	Chris Goringe 	M 	train-other-500
    108, #	Kevin McAsh 	M 	train-clean-360
    130, # 	Peter of Buckinghamshire England 	M 	train-other-500
    661, #	James Gladwin 	M 	train-other-500
    216, #	Dave Ranson 	M 	train-clean-100
    164, #	Ed Good 	M 	train-other-500
    308, #	Eric Connover 	M 	train-other-500
    569, #	Arouet 	M 	train-clean-360
    313, #	Tim Bulkeley 	M 	train-other-500
    212, #	Glen Hallstrom 	M 	train-other-500
    15,  # 	Chip 	M 	train-other-500
    469, #	Christian Pecaut 	M 	train-clean-360
    294, # 	Diana Kiesners 	F 	train-clean-360
    192, #	Nocturna 	F 	train-clean-100
    73,  #	Claire Goget 	F 	train-clean-100
    417, #	Kiki Baessell 	F 	train-clean-360
    636, #	Matthew Howell 	F 	train-other-500
    36,  #	chriss the girl 	F 	train-other-500
    668, #	Jan Baxter 	F 	train-clean-360
    403, #	Igor Teaforay 	F 	train-clean-360
    618, #	Linnea 	F 	train-other-500
    596, #	Jo 	F 	train-other-500
    499, #	Tammy Sanders 	F 	train-clean-100
    207, #	Sage Tyrtle 	F 	train-other-500
    1346, #	Jeanie 	F 	train-other-500
    1109, #	Martin Geeson 	M 	train-other-500
    770,  #	Pete Williams, Pittsburgh, PA 	M 	train-clean-360
    1247, #	Sarah LuAnn 	F 	train-clean-100
    1526, # 	Mike Harris 	M 	train-other-500
    908,  #	Quentin Manuel 	M 	train-clean-360
    1183, # 	Evelyn Clarke 	F 	train-other-500
    1438, #	Tom Barron 	M 	train-other-500
    1022, # 	peac 	M 	train-clean-100
    1603, # 	Christine Rodriguez 	F 	train-clean-360
    1425, # 	Jonah Cummings 	M 	train-clean-360
    731,  # 	Priya, India 	F 	train-other-500
    782,  #	Alec Daitsman 	M 	train-clean-360
    1090, #	Termin Dyan 	M 	train-other-500
    995,  #	Parrot 	M 	train-other-500
    923,  #	Jane Greensmith 	F 	train-clean-360
    766,  #	Clive Catterall 	M 	train-other-500
    822,  #	kristiface 	F 	train-clean-360
    897,  #	Jan Dawn Doronila 	F 	train-clean-360
    1579, #	Linda Velwest 	F 	train-clean-360
    964,  #	Utek 	M 	train-clean-360
    1414, # 	Preston Scrape 	M 	train-other-500
    834,  #	Serin 	F 	train-other-500
    1302, #	davidb 	M 	train-clean-360
    1135, #	Linda Andrus 	F 	train-clean-360
    1440, # 	P Moscato 	F 	train-clean-360
    870,  #	Barbara Bulkeley 	F 	train-clean-360
    1256, #	Graeme Dunlop 	M 	train-other-500
    1255, #	Daniel Paashaus 	M 	train-other-500
    1157, #	Bev J Stevens 	F 	train-clean-360
    934,  #	Darla 	F 	train-other-500
    1281, #	garbageman99 	M 	train-clean-360
    819,  #	n8evv 	M 	train-clean-360
    1041, #	mjbrichant 	F 	train-other-500
    863,  #	K Hindall 	F 	train-clean-360
    1303, #	kiwafruit 	F 	train-clean-100
    1115, #	Rachel Gatwood 	F 	train-clean-360
    1539, #	Nathan Jordan 	M 	train-other-500
    1428, #	Gary Dzierlenga 	M 	train-other-500
    1049, #	Diana Solomon 	F 	train-other-500
    1546, #	Carrie Heyes 	F 	train-other-500
    1089, #	Bill Ruhsam 	M 	train-clean-360
    1142, #	Jonathan Burchard 	M 	train-other-500
    1375, #	Frank Adams 	M 	train-clean-360
    881,  #	mpetranech 	M 	train-other-500
    798,  #	Wyatt 	M 	train-other-500
    1647, # 	Patrick Reinhart 	M 	train-clean-360
    1587, #	Claudia Wilson 	F 	train-clean-360
    830,  #	musici123 	F 	train-other-500
    1592, #	jerryB 	M 	train-other-500
    839,  #	Ben Dutton 	M 	train-other-500
    835,  #	Rachel Lintern 	F 	train-other-500
    1273, #	gmiteva 	F 	train-other-500
    932,  #	Raerity 	F 	train-other-500
    1108, #	Paul McCartan 	M 	train-other-500
    732,  #	Tysto 	M 	train-clean-360
    781,  #	Megan Kunkel 	F 	train-other-500
    1555, #	Andrew Nelson 	M 	train-clean-360
    1437, #	Charles RUHE 	M 	train-clean-360
    1402, #	Angel5 	F 	train-other-500
    963,  #	MichelleHarris 	F 	train-clean-360
    1181, #	J. Rebecca Franklin 	F 	train-clean-360
    818,  #	Matt Warzel 	F 	train-clean-360
    1285, #	Ric F 	M 	train-clean-100
    797,  #	Chris Jones 	F 	train-other-500
    1505, #	Rom Maczka 	M 	train-clean-360
    1214, #	David Baldwin 	M 	train-clean-360
    1636, #	jessecoy 	M 	train-other-500
    929,  #	Petra 	F 	train-other-500
    1171, # 	Roberta Carlisle 	F 	train-other-500
    817,  #	texttalker 	M 	train-clean-360
    1433, #	browneyedgirl32382 	F 	train-clean-360
    1158, #	StarrDog 	M 	train-other-500
    1000, #	artos 	M 	train-other-500
    848,  #	senshisteph 	F 	train-other-500
    1596, #	Joyce Couch 	F 	train-other-500
    757,  #	Roger Melin 	M 	train-clean-360
    1168, #	Epistomolus 	M 	train-clean-100
    741,  #	Nick Marsh 	M 	train-other-500
    1649, #	Phineas Redux 	M 	train-other-500
    851,  #	Jennifer Lott 	F 	train-clean-360
    808,  #	M. J. Boyle 	F 	train-other-500
    1595, #	Matthew Reece 	M 	train-clean-360
    1370, #	Savanna Herrold 	F 	train-other-500
    1565, #	bryan.peterson 	M 	train-other-500
    944,  #	Sarafina Suransky 	F 	train-other-500
    1268, #	A. Janelle Risa 	F 	train-clean-100
    771,  #	Isosceles 	F 	train-clean-360
    752,  #	Cat Schirf 	F 	train-other-500
    800,  #	Jack Farrell 	M 	train-clean-360
    1005, #	Beatrice 	F 	train-other-500
    1229, #	RoseA 	F 	train-clean-360
    943,  #	Matthew C. Heckel 	M 	train-clean-360
    891,  #	anoldfashiongirl 	F 	train-other-500
    1226, #	serenitylee 	F 	train-clean-360
    1253, #	Caroline Shapiro 	F 	train-other-500
    1204, #	Dale A. Bade 	F 	train-clean-360
    1230, #	Troy Bond 	M 	train-other-500
    791,  #	David Kleparek 	M 	train-clean-100
    1184, #	Joseph Couves 	F 	train-other-500
    1001, #	TriciaG 	F 	train-clean-360
    804,  #	FirstKnight 	F 	train-other-500
    1641, #	Kirsten Wever 	F 	train-clean-100
    1259, # 	Megan Argo 	F 	train-other-500
    1231, #	Abigail Bartels 	F 	train-other-500
    1410, # 	Zachary Johnson 	M 	train-other-500
    1030, #	Ancient mariner 	M 	train-other-500
    1093, #	Katie Riley 	F 	train-clean-360
    1254, #	Rosie 	F 	train-clean-100
    1365, #	Eric Leach 	M 	train-clean-360
    831,  #	David Federman 	M 	train-other-500
    1989, # 	Joannemmp 	F 	train-clean-100
    1707, #	David Olson 	M 	train-other-500
    1849, #	Fred DeBerardinis 	M 	train-clean-100
    1808, #	Rebecca King 	F 	train-clean-360
    2292, #	Arnold 	M 	train-clean-100
    2415, #	Patrick Eaton 	M 	train-other-500
    1656, #	Sharon Omi 	F 	train-clean-100
    1676, #	Gargoyle 	M 	train-clean-360
    1881, #	Julienne 	F 	train-other-500
    2036, #	T.K. Kirven 	F 	train-other-500
    1761, #	EliMarieHK 	F 	train-other-500
    2115, #	Pete Milan 	M 	train-other-500
    1803, #	Susan Hanfield 	F 	train-clean-360
    1798, #	C. L. W. Rollins 	F 	train-other-500
    1723, #	Rachel Bossier 	F 	train-other-500
    2341, #	Haili 	F 	train-other-500
    2468, #	Erin Schellhase 	F 	train-clean-360
    1725, #	Ruth Kidson 	F 	train-other-500
    2010, #	Peggy 	F 	train-other-500
    1853, #	Ron Altman 	M 	train-other-500
    2359, #	Doug Reed 	M 	train-other-500
    2422, #	Jude Somers 	F 	train-clean-360
    2234, #	Coreena 	F 	train-other-500
    2156, # 	C F de Rosset 	F 	train-other-500
    2483, #	Tammy Porter 	F 	train-clean-360
    1781, #	humanode 	M 	train-clean-360
    2275, #	NatalieOram 	F 	train-other-500
    2390, #	sdaeley17 	M 	train-clean-360
    2314, #	Cheri Jordan 	F 	train-clean-360
    2413, #	Joanne Rochon 	F 	train-clean-360
    1697, # 	Lonelle Yoder 	F 	train-other-500
    1718, # 	Caroline Driggs 	F 	train-other-500
    2387, #	Brett G. Hirsch 	M 	train-other-500
    2331, #	Madam Fickle 	F 	train-clean-100
    1783, #	Sarah Crampton 	F 	train-clean-360
    2397, #	Rebecca Braunert-Plunkett 	F 	train-other-500
    2357, #	William Gavula 	M 	train-other-500
    1670, #	dmbrought 	M 	train-other-500
    1987, #	Andrew White 	M 	train-clean-360
    1755, # 	Yvonne Smith 	F 	train-clean-360
    2192, #	Sammy Bean 	M 	train-other-500
    1716, #	EyeBones 	F 	train-clean-360
    1828, #	David Wales 	M 	train-clean-100
    2251, #	Wiley Combs 	M 	train-clean-360
    2065, #	Muriel 	F 	train-clean-360
    2017, #	CaprishaPage 	F 	train-other-500
    1947, #	Barbara Edelman 	F 	train-other-500
    1738, #	Lois C. Johnson 	F 	train-clean-360
    1791, #	David Cummings 	M 	train-clean-360
    2045, #	Linda Ciano 	F 	train-clean-360
    2452, # 	Walt Allan 	M 	train-other-500
    2040, #	MJ Franck 	F 	train-other-500
    1831, #	Nigel Boydell 	M 	train-other-500
    2371, #	Alexander Hatton 	M 	train-clean-360
    1954, #	Szindbad 	M 	train-other-500
    1836, #	Kendall Ashyby 	F 	train-other-500
    2436, # 	josembi 	M 	train-other-500
    2383, # 	Emma Joyce 	F 	train-other-500
    2278, #	Jake Woldstad 	M 	train-clean-360
    1741, # 	anjieliu 	F 	train-other-500
    1857, #	Amanda Friday 	F 	train-clean-360
    2370, # 	gloriousjob 	M 	train-clean-360
    1907, # 	Snapdragon 	F 	train-other-500
    2225, # 	nomorejeffs 	M 	train-clean-360
    2439, #	KHand 	F 	train-clean-360
    2239, #	amaskill 	M 	train-other-500
    2007, #	Art Leung 	F 	train-clean-360
    2283, #	Tim Cote 	M 	train-clean-360
    1712, # 	Steve Belleguelle 	M 	train-other-500
    2094, #	Meg Cowan 	F 	train-clean-360
    1772, # 	haggisreflux 	M 	train-clean-360
    2317, # 	helengraves 	F 	train-clean-360
    2241, # 	Steven Reynolds 	M 	train-clean-360
    2011, # 	pekein 	M 	train-clean-360
    1826, # 	John Hoerr 	M 	train-clean-100
    1695, #	Tina Nuzzi 	F 	train-clean-360
    2451, #	DeanOBuchanan 	M 	train-clean-100
    1771, #	Chelsea S. 	F 	train-other-500
    2441, #	Alison Stewart 	F 	train-clean-360
    1745, #	Janet 	F 	train-clean-360
    2358, # 	Betty Perry 	F 	train-clean-360
    2197, #	Mike Nelson 	M 	train-other-500
    2014, # 	Eden Rea-Hedrick 	F 	train-other-500
    1672, # 	Mike Wajda 	M 	train-clean-360
    2394, #	TinaNygard2 	F 	train-clean-100
    1657, #	alwpoe 	M 	train-clean-360
    1728, #	Vinnie Tesla 	M 	train-clean-360
    1805, # 	Vince Dee 	M 	train-clean-100
    2143, # 	Suebee 	F 	train-clean-360
    2084, #	Eberle Thomas 	M 	train-other-500
    2479, #	Daisy Flaim 	F 	train-clean-100
    2152, #	Kristel Tretter 	F 	train-clean-360
    2268, #	Greg Giordano 	M 	train-clean-360
    1839, #	James E. Carson 	M 	train-clean-360
    2056, # 	acloward 	M 	train-clean-360
    1814, #	polkadotish 	F 	train-other-500
    2127, #	Ron Lockhart 	M 	train-clean-100
    2114, #	Larry Beasley 	M 	train-clean-360
    2469, # 	Kevin Owens 	M 	train-clean-100
    2447, #	Deena Rhoads 	F 	train-clean-360
    1724, #	Juliana M. 	F 	train-clean-360
    1869, # 	NastassiaS 	F 	train-other-500
    2209, #	Samantha J Gubitz 	F 	train-clean-360
    2171, # 	Carolyne 	F 	train-other-500
    2403, #	Ian Quinlan 	M 	train-clean-360
    2032, # 	doonaboon 	M 	train-other-500
    2075, #	Joy S Grape 	F 	train-clean-360
]

# Convert the model speaker IDs back to the dataset speaker IDs
dataset_speaker_ids: List[int] = [
    reverse_mapping.get(int(speaker_id))
    for speaker_id in selected_speakers
] # type: ignore
