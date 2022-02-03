def move_gripper(opened):
	service = '/kair_evg55_2/open'
	rospy.wait_for_service(service)
	try:
		moveGripper = rospy.ServiceProxy(service, Open)
		return moveGripper(50 if opened else 0)
	except rospy.ServiceException as e:
		print('Service call failed: %s'%e)
