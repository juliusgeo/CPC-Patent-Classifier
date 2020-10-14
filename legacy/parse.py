import os
from lxml import etree
from bs4 import BeautifulSoup
parser = etree.XMLParser(remove_blank_text=True)
files = os.listdir("data")
files = ["ipg191001.xml"]
print(files)
import csv
for filename in files:
	with open(os.path.join("data", filename), 'r') as f:
		grants = []
		try:
			for line in f:
				if line == "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n":
					try:
						grants.append(parser.close())
					except:
						pass
				else:
					try:
						parser.feed(line)
					except:
						pass
			try:
				grants.append(parser.close())
			except:
				pass
			with open('test_dataset.csv', mode='a') as file:
				class_and_text = [(set(z.text for z in i.iter('classification-cpc-text')), [etree.tostring(z, encoding='utf-8', xml_declaration=False, pretty_print=True, standalone='yes') for z in i.iter('abstract')]) for i in grants]
				writer = csv.writer(file, delimiter=',', quotechar="\"", quoting=csv.QUOTE_MINIMAL)
				for grant in class_and_text:
					try:
						if grant[1] == [] or grant[0] == set():
							continue
						abstract_html = b" ".join(grant[1])
						soup = BeautifulSoup(abstract_html, features='lxml')
						abstract_text = " ".join(soup.strings)
						writer.writerow(["\""+','.join(list(grant[0]))+"\"", "\""+str(abstract_text.strip())+"\""])
					except:
						pass
		except:
			pass


print("Number of grants: "+str(len(class_and_text)))

