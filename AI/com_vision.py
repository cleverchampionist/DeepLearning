import cv2

image  = cv2.imread('poland1.jpg')
image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('edges_Penguins.jpg', cv2.Canny(image, 200, 300))

cv2.imshow('poland man', cv2.imread('edges_Penguins.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()