
            current_perspective = "back"
    else:
        # Reset video capture to start from the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Release the video capture object and close the d