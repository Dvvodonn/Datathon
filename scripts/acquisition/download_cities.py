import urllib.request, zipfile, io, json, csv

# Download Spain GeoNames dump (public domain)
url = "http://download.geonames.org/export/dump/ES.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
    data = z.read("ES.txt").decode("utf-8")

features = []
for row in csv.reader(io.StringIO(data), delimiter="\t"):
    if row[6] == "P" and int(row[14] or 0) >= 2000:  # P = populated place
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row[5]), float(row[4])]},
            "properties": {"name": row[1], "population": int(row[14])}
        })

with open("spain_cities_full.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)