import cv2
import torch

class Segmenter:
    def __init__(self, model_path, size, save = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path,map_location=self.device).eval()
        self.size = size
        self.save  = save 

    def __call__(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.size,self.size))
        transformed_image = torch.Tensor(image)
        transformed_image = transformed_image.permute((2, 0, 1))
        transformed_image /= 255
        transformed_image.to(self.device)
        self.model.eval()
        print(transformed_image.unsqueeze(0).shape)
        result = self.model(transformed_image.unsqueeze(0))
        return image,result

class Points_finder:
    def __init__(self, min_contour_length = 50,threshold_border = 125, save = False):
        self.threshold_border = threshold_border
        self.min_contour_length = min_contour_length 
        self.save = save

    def __call__(self, segmented_image):
        _, binary_image = cv2.threshold(segmented_image * 255, 125, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_length = 50 

        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
        starting_points = [self.find_topmost_point(contour) for contour in filtered_contours]

        return starting_points,filtered_contours

    def find_topmost_point(self,contour):
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        return topmost
    
    def plot(self, segmented_image, original_image):
        points,contours = self.__call__(segmented_image)
        image = original_image.copy()
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Blue contours

        for point in points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green starting points

        return image,points
