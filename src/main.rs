fn greet(name: &str) {
    println!("Hello {}!", name);
}

fn fair_dice_roll() -> i32 {
    4
}

struct Vec2 {
    x: f64,
    y: f64,
}

struct Zahl {
    ungerade: bool,
    wert: i32,
}

impl Zahl {
    fn is_strictly_positive(self) -> bool {
        self.wert > 0
    }
}

trait Signed {
    fn is_strictly_negative(self) -> bool;
}

impl Signed for Zahl {
    fn is_strictly_negative(self) -> bool {
        self.wert < 0
    }
}

fn print_zahl(z: &Zahl) {
    match z.wert {
        1 => println!("Eins"),
        2 => println!("Zwei"),
        _ => println!("{}", z.wert),
    }
}

fn file_ext(name: &str) -> Option<&str> {
    // this does not create a new string - it returns
    // a slice of the argument.
    name.split(".").last()
}

fn main() {
    let emre = String::from("Emre");
    let x = vec![1, 2, 3].iter().map(|x| x + 3).fold(0, |x, y| x + y);

    println!("The sum is {}", x);
    greet(&emre);
    greet(&emre);

    let v1 = Vec2 { x: 1.0, y: 3.0 };

    let one = Zahl {
        ungerade: true,
        wert: 1,
    };
    let two = Zahl {
        ungerade: false,
        wert: 2,
    };

    let minus_two = Zahl {
        ungerade: false,
        wert: -2,
    };

    print_zahl(&one);
    print_zahl(&two);
    println!("negative? {}", minus_two.is_strictly_negative());

    let name = String::from("Read me. Or don't.txt");
    let ext = { file_ext(&name).unwrap_or("") };
    println!("extension: {:?}", ext);
}
