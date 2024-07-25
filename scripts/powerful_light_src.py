import cv2

with_light = cv2.imread('./images/light_sources/with_light.jpg')
without_light_1 = cv2.imread('./images/light_sources/without_light_1.jpg')
without_light_2 = cv2.imread('./images/light_sources/without_light_2.jpg')

with_light_gray = cv2.cvtColor(with_light, cv2.COLOR_BGR2GRAY)
without_light_1_gray = cv2.cvtColor(without_light_1, cv2.COLOR_BGR2GRAY)
without_light_2_gray = cv2.cvtColor(without_light_2, cv2.COLOR_BGR2GRAY)

with_light_gray_mean = with_light_gray.mean()
without_light_gray_mean_1 = without_light_1_gray.mean()
without_light_gray_mean_2 = without_light_2_gray.mean()

std1 = with_light_gray.std()
std2 = without_light_1_gray.std()
std3 = without_light_2_gray.std()

print(f"Mean value of image that has powerful light is {with_light_gray_mean}")
print(f"Mean value of first image that has no powerful light is {without_light_gray_mean_1}")
print(f"Mean value of second image that has no powerful light is {without_light_gray_mean_2}")
print(f"Sandard1: {std1}")
print(f"Sandard2: {std2}")
print(f"Sandard3: {std3}")
