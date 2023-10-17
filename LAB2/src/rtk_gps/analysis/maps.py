import rospy
from nav_msgs.msg import Odometry
from gmplot import gmplot
from rosbag import Bag

# Read the latitudes and longitudes from the bag file
def read_lat_lng_from_bag(bagfile):
    latitude = []
    longitude = []

    # Initialize ROS node (make sure roscore is running)
    rospy.init_node('bag_reader', anonymous=True)
    
    bag = Bag(bagfile, 'r')
    for topic, msg, _ in bag.read_messages(topics=['/GPS']):
        lat = msg.latitude
        lng = msg.longitude
        latitude.append(lat)
        longitude.append(lng)

    bag.close()

    return latitude, longitude

def main():
    bagfile = 'walking_data_occ.bag'
    latitude, longitude = read_lat_lng_from_bag(bagfile)
    
    # Get the first latitude and longitude for centering the map
    gmap = gmplot.GoogleMapPlotter(latitude[0], longitude[0], 16)
    
    # Plot the points
    gmap.scatter(latitude, longitude, '#FF0000', size=2, marker=False)
    
    # Save to an HTML file
    gmap.draw('map3.html')

if __name__ == "__main__":
    main()
 
