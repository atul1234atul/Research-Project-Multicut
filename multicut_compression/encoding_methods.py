"""
Multiple Encoding Methods for RGB Region Values

This module implements different methods for encoding RGB values of image regions,
as required by the task: "Different methods for encoding the RGB values of 
image regions should be tested."


"""

import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter
import zlib
import struct
import io


class RegionEncoder:
    """Base class for region encoding methods"""
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode pixel values"""
        raise NotImplementedError
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode pixel values"""
        raise NotImplementedError
    
    def calculate_bits(self, pixels: List[Any]) -> int:
        """Calculate number of bits needed to encode pixels"""
        encoded = self.encode(pixels)
        return len(encoded) * 8


class SimpleRegionEncoder(RegionEncoder):
    """Simple encoding: store most common pixel value"""
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode using most common pixel value"""
        if not pixels:
            return b''
        
        # Find most common pixel value
        pixel_counts = Counter(pixels)
        most_common_pixel = pixel_counts.most_common(1)[0][0]
        count = len(pixels)
        
        # Encode as: [value, count]
        if isinstance(most_common_pixel, tuple):
            # RGB - convert to single integer representation
            encoded_value = (most_common_pixel[0] << 16) | (most_common_pixel[1] << 8) | most_common_pixel[2]
        else:
            encoded_value = int(most_common_pixel)
        
        # Convert to bytes
        byte_data = bytearray()
        byte_data.extend(struct.pack('>I', encoded_value))  # 4 bytes for value
        byte_data.extend(struct.pack('>I', count))  # 4 bytes for count
        
        return bytes(byte_data)
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode pixel values"""
        if len(encoded_data) < 8:
            return []
        
        encoded_value = struct.unpack('>I', encoded_data[0:4])[0]
        count = struct.unpack('>I', encoded_data[4:8])[0]
        
        # Extract RGB values
        r = (encoded_value >> 16) & 0xFF
        g = (encoded_value >> 8) & 0xFF
        b = encoded_value & 0xFF
        pixel_value = (r, g, b)
        
        return [pixel_value] * count


class HuffmanRegionEncoder(RegionEncoder):
    """Huffman coding for region encoding"""
    
    def __init__(self):
        self.code_map = {}
    
    def _build_huffman_tree(self, data: List[Any]) -> Dict[Any, str]:
        """Build Huffman tree"""
        if not data:
            return {}
        
        # Convert pixels to comparable format
        pixel_to_int = {}
        int_to_pixel = {}
        int_data = []
        
        for pixel in data:
            if pixel not in pixel_to_int:
                idx = len(pixel_to_int)
                pixel_to_int[pixel] = idx
                int_to_pixel[idx] = pixel
            int_data.append(pixel_to_int[pixel])
        
        # Count frequencies
        freq = Counter(int_data)
        
        # Build Huffman tree (simplified)
        import heapq
        heap = []
        for value, count in freq.items():
            heapq.heappush(heap, (count, value))
        
        codes = {}
        if len(heap) == 1:
            codes[heap[0][1]] = '0'
        else:
            while len(heap) > 1:
                count1, value1 = heapq.heappop(heap)
                count2, value2 = heapq.heappop(heap)
                # Create internal node
                new_value = (value1, value2)
                heapq.heappush(heap, (count1 + count2, new_value))
            
            if heap:
                _, tree = heap[0]
                self._generate_codes(tree, '', codes, int_to_pixel)
        
        return codes
    
    def _generate_codes(self, tree: Any, prefix: str, codes: Dict, int_to_pixel: Dict):
        """Recursively generate Huffman codes"""
        if isinstance(tree, int):
            codes[int_to_pixel[tree]] = prefix
        elif isinstance(tree, tuple):
            left, right = tree
            self._generate_codes(left, prefix + '0', codes, int_to_pixel)
            self._generate_codes(right, prefix + '1', codes, int_to_pixel)
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode using Huffman coding"""
        if not pixels:
            return b''
        
        codes = self._build_huffman_tree(pixels)
        
        # Encode data
        bit_string = ''.join(codes[pixel] for pixel in pixels)
        
        # Pad to byte boundary
        while len(bit_string) % 8 != 0:
            bit_string += '0'
        
        # Convert to bytes
        byte_array = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_array.append(int(bit_string[i:i+8], 2))
        
        # Store code map
        encoded_data = bytearray()
        encoded_data.extend(len(codes).to_bytes(2, 'big'))
        
        for pixel, code in codes.items():
            # Store pixel value
            if isinstance(pixel, tuple):
                encoded_data.extend(struct.pack('>BBB', *pixel))
            else:
                encoded_data.append(int(pixel))
            # Store code length and code
            encoded_data.extend(len(code).to_bytes(1, 'big'))
            code_bytes = int(code, 2).to_bytes((len(code) + 7) // 8, 'big')
            encoded_data.extend(code_bytes)
        
        encoded_data.extend(byte_array)
        return bytes(encoded_data)
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode Huffman-encoded data"""
        # Implementation would decode the code map and data
        # Simplified for now
        return []


class RLERegionEncoder(RegionEncoder):
    """Run-Length Encoding for region encoding"""
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode using Run-Length Encoding"""
        if not pixels:
            return b''
        
        encoded = bytearray()
        i = 0
        
        while i < len(pixels):
            current_pixel = pixels[i]
            count = 1
            
            # Count consecutive identical pixels
            while i + count < len(pixels) and pixels[i + count] == current_pixel:
                count += 1
            
            # Encode pixel value
            if isinstance(current_pixel, tuple):
                encoded.extend(struct.pack('>BBB', *current_pixel))
            else:
                encoded.append(int(current_pixel))
            
            # Encode count
            encoded.extend(struct.pack('>I', count))
            
            i += count
        
        return bytes(encoded)
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode RLE-encoded data"""
        pixels = []
        i = 0
        
        while i < len(encoded_data):
            if i + 7 > len(encoded_data):  # Need at least 7 bytes (3 for RGB + 4 for count)
                break
            
            # Decode pixel value
            r = encoded_data[i]
            g = encoded_data[i + 1]
            b = encoded_data[i + 2]
            pixel_value = (r, g, b)
            
            # Decode count
            count = struct.unpack('>I', encoded_data[i + 3:i + 7])[0]
            
            pixels.extend([pixel_value] * count)
            i += 7
        
        return pixels


class DeltaRegionEncoder(RegionEncoder):
    """Delta encoding for region encoding"""
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode using delta encoding"""
        if not pixels:
            return b''
        
        # Convert to flat array
        flat_pixels = []
        for pixel in pixels:
            if isinstance(pixel, tuple):
                flat_pixels.extend(pixel)
            else:
                flat_pixels.append(int(pixel))
        
        # Calculate deltas
        deltas = [flat_pixels[0]]  # First value
        for i in range(1, len(flat_pixels)):
            deltas.append(flat_pixels[i] - flat_pixels[i-1])
        
        # Compress deltas with zlib
        delta_bytes = struct.pack('>' + 'b' * len(deltas), *deltas)
        compressed = zlib.compress(delta_bytes)
        
        return compressed
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode delta-encoded data"""
        # Decompress
        delta_bytes = zlib.decompress(encoded_data)
        deltas = struct.unpack('>' + 'b' * (len(delta_bytes)), delta_bytes)
        
        # Reconstruct original values
        flat_pixels = [deltas[0]]
        for i in range(1, len(deltas)):
            flat_pixels.append(flat_pixels[i-1] + deltas[i])
        
        # Convert back to pixels
        pixels = []
        for i in range(0, len(flat_pixels), 3):
            if i + 2 < len(flat_pixels):
                pixels.append((flat_pixels[i], flat_pixels[i+1], flat_pixels[i+2]))
        
        return pixels


class ZlibRegionEncoder(RegionEncoder):
    """Zlib compression for region encoding (current method)"""
    
    def encode(self, pixels: List[Any]) -> bytes:
        """Encode using zlib compression"""
        if not pixels:
            return b''
        
        # Convert pixels to bytes
        pixel_bytes = bytearray()
        for pixel in pixels:
            if isinstance(pixel, tuple):
                pixel_bytes.extend(struct.pack('>BBB', *pixel))
            else:
                pixel_bytes.append(int(pixel))
        
        # Compress with zlib
        compressed = zlib.compress(bytes(pixel_bytes), level=9)
        return compressed
    
    def decode(self, encoded_data: bytes) -> List[Any]:
        """Decode zlib-compressed data"""
        # Decompress
        pixel_bytes = zlib.decompress(encoded_data)
        
        # Convert back to pixels
        pixels = []
        for i in range(0, len(pixel_bytes), 3):
            if i + 2 < len(pixel_bytes):
                pixels.append((pixel_bytes[i], pixel_bytes[i+1], pixel_bytes[i+2]))
        
        return pixels


def get_all_encoders() -> Dict[str, RegionEncoder]:
    """Get all available encoders"""
    return {
        'simple': SimpleRegionEncoder(),
        'huffman': HuffmanRegionEncoder(),
        'rle': RLERegionEncoder(),
        'delta': DeltaRegionEncoder(),
        'zlib': ZlibRegionEncoder()
    }







