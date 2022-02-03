def pcl_callback(msg):
	i = 0
	x_values = []
	y_values = []
	z_values = []
  
	while i < len(msg.data):
		point = msg.data[i * msg.point_step:(i+1) * msg.point_step]
		
		if len(point) >= 20:
			x = struct.unpack('f', point[0:4])[0]
			y = struct.unpack('f', point[4:8])[0]
			z = struct.unpack('f', point[8:12])[0]
			
			x_values.append(x)
			y_values.append(y)
			z_values.append(z)
			
		i += msg.point_step
		
	A = np.zeros((len(x_values), 3))
	A[:, 0] = x_values[:]
	A[:, 1] = y_values[:]
	A[:, 2] = z_values[:]
	
  with open('brick.npy', 'wb') as file:
    np.save(file, A)

rospy.init_node('rs_pcl')
rospy.Subscriber('/camera/depth/color/points', 
                 PointCloud2, pcl_callback)

while not rospy.is_shutdown():
	pass