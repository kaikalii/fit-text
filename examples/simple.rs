use fit_text::*;
use graphics_buffer::*;

fn main() {
    // The rectangle in which the text will be fit
    let rect = [0.0, 0.0, 200.0, 200.0];
    // A basic string
    let s1 = "this is a test string with enough words to go onto another line or two";
    // String fragments with associated colors
    let s2 = vec![
        ("this is a test string", [1.0, 0.0, 0.0, 1.0]),
        ("with enough words", [0.0, 1.0, 0.0, 1.0]),
        ("to go onto", [1.0, 1.0, 0.0, 1.0]),
        ("another line or two", [0.0, 0.0, 1.0, 1.0]),
    ];
    // Initialize glyphs
    let mut glyphs = buffer_glyphs_from_bytes(include_bytes!("roboto.ttf")).unwrap();

    // Initialize two render buffers
    let mut normal_buffer = RenderBuffer::new(200, 200);
    let mut colored_buffer = RenderBuffer::new(200, 200);

    // Draw normal text
    fitted_text(s1, rect, 30, &mut glyphs, IDENTITY, &mut normal_buffer).unwrap();

    // Draw colored text
    fitted_colored_text(s2, rect, 30, &mut glyphs, IDENTITY, &mut colored_buffer).unwrap();

    // Save buffers
    normal_buffer.save("normal.png").unwrap();
    colored_buffer.save("colored.png").unwrap();
}
