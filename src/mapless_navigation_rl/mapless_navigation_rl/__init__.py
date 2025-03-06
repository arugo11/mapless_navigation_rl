# visualization_msgsをインポート 
from visualization_msgs.msg import Marker

# __init__メソッド内
# 目標マーカーのパブリッシャー
self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
# 定期的に目標マーカーを発行するタイマー
self.marker_timer = self.create_timer(1.0, self.publish_goal_marker)