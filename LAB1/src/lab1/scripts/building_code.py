import utm

with open("walking_data1.txt") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith("$GPGGA"):
            fields = line.split(',')
            if len(fields) >= 10:
                utc = fields[1]
                lat = float(fields[2])/100
                lon = float(fields[4])/100
                alt = fields[9]

                # Convert lat and lon to UTM format
                easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)

                print("UTC:", utc)
                print("Latitude (NMEA):", lat)
                print("Longitude (NMEA):", lon)
                print("Altitude:", alt)
                print("Easting (UTM):", easting)
                print("Northing (UTM):", northing)
                print("Zone Number:", zone_number)
                print("Zone Letter:", zone_letter)
