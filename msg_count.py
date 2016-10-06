import rosbag
import sys

if __name__ == '__main__':
    topic = sys.argv[1]
    bagFile = sys.argv[2]
    bag = rosbag.Bag(bagFile, 'r')
    print bag.get_message_count(topic) 
