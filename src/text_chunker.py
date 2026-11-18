
def split_text_into_chunks(
    text,
    chunk_size=1000,
    overlap=150
):
    """
    Split text into chunks while preserving complete lines.

    Chunk size and overlap are measured in characters.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:

        line_length = len(line)

        # If adding a line exceeds the chunk size,
        # save the current chunk first.
        if current_lines and current_length + line_length + 1 > chunk_size:

            chunks.append("\n".join(current_lines))

            # Preserve complete trailing lines as overlap.
            overlap_lines = []
            overlap_length = 0

            for previous_line in reversed(current_lines):

                required_length = len(previous_line) + 1

                if overlap_length + required_length > overlap:
                    break

                overlap_lines.insert(0, previous_line)
                overlap_length += required_length

            current_lines = overlap_lines
            current_length = sum(
                len(item) + 1 for item in current_lines
            )

        current_lines.append(line)
        current_length += line_length + 1

    # Save remaining content.
    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def chunk_medical_pages(
    pages,
    chunk_size=1000,
    overlap=150
):
    """
    Split processed medical pages into chunks.
    Preserve page numbers and chunk metadata.
    """

    all_chunks = []
    chunk_id = 1

    for page in pages:

        page_chunks = split_text_into_chunks(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk_text in enumerate(page_chunks):

            all_chunks.append({

                "chunk_id": chunk_id,

                "page_number": page["page_number"],

                "chunk_index": index,

                "text": chunk_text

            })

            chunk_id += 1

    return all_chunks
