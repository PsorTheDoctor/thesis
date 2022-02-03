def move(array):
	service = '/ur5_1/move_ptp_p'
	
	if len(array) == 6:
		rospy.wait_for_service(service)
		try:
			movePtp = rospy.ServiceProxy(service, MovePTP_P)
			req = MovePTP_PRequest()
			req.target.data = array
			req.a = 1
			req.v = 1
			req.t = 5
			req.r = 0
			resp = movePtp(req)
			return resp
		except rospy.ServiceException as e:
			print('Service call failed: %s'%e)
	else:
		print('Array length must be 6')