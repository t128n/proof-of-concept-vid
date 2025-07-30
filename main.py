import cv2
import numpy as np
import os
import whisper
import subprocess
import ollama
import concurrent.futures

def delete_similar_frames(out_dir='./out', similarity_threshold=0.95):
    """Delete frames in out_dir that are very similar to their previous frame."""
    jpg_files = sorted([fn for fn in os.listdir(out_dir) if fn.startswith("frame_") and fn.endswith(".jpg")])
    prev_img = None
    for filename in jpg_files:
        img_path = os.path.join(out_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev_img is not None:
            # Compare similarity
            diff = cv2.absdiff(prev_img, img)
            similarity = 1.0 - (np.sum(diff) / (255 * diff.size))
            if similarity >= similarity_threshold:
                os.remove(img_path)
                print(f"Deleted similar frame: {filename}")
                continue
        prev_img = img

def summarize_video_with_gemma(out_dir='./out', model="gemma3"):
    """Prompt Gemma-3 with all image descriptions and transcript.txt to summarize the video."""
    # Collect transcript
    transcript_path = os.path.join(out_dir, 'transcript.txt')
    transcript = ""
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
    # Collect all frame descriptions
    descriptions = []
    for filename in sorted(os.listdir(out_dir)):
        if filename.startswith("frame_") and filename.endswith(".txt"):
            desc_path = os.path.join(out_dir, filename)
            with open(desc_path, 'r', encoding='utf-8') as f:
                descriptions.append(f"{filename}:\n" + f.read())
    # Compose prompt
    prompt = "Summarize the following video based on its transcript and frame descriptions.\n\nTranscript:\n" + transcript + "\n\nFrame Descriptions:\n" + "\n\n".join(descriptions)
    # Send to Ollama
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    summary = response["response"]
    # Save summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Video summary saved to {summary_path}")

def describe_image_with_gemma(image_path, model="gemma3"):
    # Read image as bytes
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    # Send to Ollama for description
    response = ollama.generate(
        model=model,
        prompt="Describe this image in detail.",
        images=[image_bytes]
    )
    return response["response"]

def process_frames_with_gemma(out_dir='./out', model="gemma3"):
    jpg_files = sorted([fn for fn in os.listdir(out_dir) if fn.startswith("frame_") and fn.endswith(".jpg")])

    def process_one(filename):
        image_path = os.path.join(out_dir, filename)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        try:
            description = ollama.generate(
                model=model,
                prompt="Describe this image in a short, concise, high-signal, low-noise way.",
                images=[open(image_path, "rb").read()]
            )["response"]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(description)
            print(f"Description saved to {txt_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_one, jpg_files)

# Call process_frames_with_gemma() after frame extraction

def clear_out_dir(out_dir='./out'):
    """Remove all files in the output directory."""
    import shutil
    if os.path.exists(out_dir):
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def extract_audio(video_path, audio_path):
    # Extract audio using ffmpeg
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, task="transcribe", word_timestamps=True)
    # Save segments with timestamps
    lines = []
    for segment in result.get("segments", []):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        lines.append(f"[{start:.2f} - {end:.2f}] {text}")
    return "\n".join(lines)


def frame_diff_percentage_per_second(video_path: str, threshold: int = 30) -> list[float]:
    """
    Calculates the percentage difference between consecutive frames,
    processing approximately one frame per second.

    Args:
        video_path: Path to the MP4 video file.
        threshold: Pixel intensity difference threshold to consider a pixel "changed".

    Returns:
        A list of percentage differences for each consecutive frame pair.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not get FPS. Defaulting to 1 frame per second.")
        frames_to_skip = 1 # Fallback if FPS not available
    else:
        # Calculate how many frames to skip to get approximately 1 frame per second
        # If FPS is 30, we read a frame, then skip 29 frames to get the next second's frame.
        frames_to_skip = int(round(fps))

    differences = []
    frame_count = 0
    prev_gray = None
    out_dir = './out'
    os.makedirs(out_dir, exist_ok=True)

    frame_save_idx = 1
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_count == 0:
            # First frame, initialize prev_gray
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame_count % frames_to_skip == 0:
            # This frame is approximately one second after the previous processed frame
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, current_gray)

            # Count pixels where difference exceeds the threshold
            changed_pixels = np.sum(diff > threshold)

            # Calculate total pixels
            total_pixels = diff.size

            # Calculate percentage difference
            percentage = (changed_pixels / total_pixels) * 100
            differences.append(percentage)

            if percentage >= threshold:
                # Save the frame to ./out/
                out_path = os.path.join(out_dir, f"frame_{frame_save_idx:04d}.jpg")
                cv2.imwrite(out_path, frame)
                frame_save_idx += 1

            prev_gray = current_gray # Update prev_gray for the next iteration

        frame_count += 1

    cap.release()
    return differences

if __name__ == "__main__":
    video_file = "input.mp4"
    audio_file = "input.wav"

    clear_out_dir()

    try:
        extract_audio(video_file, audio_file)
        transcript = transcribe_audio(audio_file)
        transcript_path = os.path.join('./out', 'transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Transcription saved to {transcript_path}")
    except Exception as e:
        print(f"Transcription error: {e}")

    try:
        diffs = frame_diff_percentage_per_second(video_file, threshold=30)
        print("\nPercentage difference for video:")
        for i, diff_percent in enumerate(diffs):
            if diff_percent < 30: 
                continue
            print(f"    Frame {i+1} vs Frame {i+2}: {diff_percent:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

    delete_similar_frames()
    process_frames_with_gemma()
    summarize_video_with_gemma()