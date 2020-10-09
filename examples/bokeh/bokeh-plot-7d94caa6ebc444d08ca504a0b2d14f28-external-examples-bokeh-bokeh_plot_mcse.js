(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("8d7d6506-0758-4310-9147-5a22a12f630b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '8d7d6506-0758-4310-9147-5a22a12f630b' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"e52fe6cb-5b67-4c44-a799-346f6c42eeb7":{"roots":{"references":[{"attributes":{"overlay":{"id":"21896"}},"id":"21890","type":"BoxZoomTool"},{"attributes":{},"id":"21930","type":"SaveTool"},{"attributes":{"callback":null},"id":"21931","type":"HoverTool"},{"attributes":{},"id":"21929","type":"UndoTool"},{"attributes":{"overlay":{"id":"21932"}},"id":"21926","type":"BoxZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21924"},{"id":"21925"},{"id":"21926"},{"id":"21927"},{"id":"21928"},{"id":"21929"},{"id":"21930"},{"id":"21931"}]},"id":"21934","type":"Toolbar"},{"attributes":{"callback":null},"id":"21895","type":"HoverTool"},{"attributes":{},"id":"21986","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"21966"}},"id":"21968","type":"CDSView"},{"attributes":{},"id":"21984","type":"BasicTickFormatter"},{"attributes":{},"id":"21925","type":"PanTool"},{"attributes":{},"id":"21894","type":"SaveTool"},{"attributes":{},"id":"21889","type":"PanTool"},{"attributes":{},"id":"21891","type":"WheelZoomTool"},{"attributes":{},"id":"21972","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"21897"}},"id":"21892","type":"LassoSelectTool"},{"attributes":{},"id":"21893","type":"UndoTool"},{"attributes":{},"id":"21908","type":"DataRange1d"},{"attributes":{"toolbar":{"id":"21996"},"toolbar_location":"above"},"id":"21997","type":"ToolbarBox"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21932","type":"BoxAnnotation"},{"attributes":{},"id":"21990","type":"UnionRenderers"},{"attributes":{"text":"mu"},"id":"21969","type":"Title"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21974"},"ticker":{"id":"21881"}},"id":"21880","type":"LinearAxis"},{"attributes":{},"id":"21991","type":"Selection"},{"attributes":{"toolbars":[{"id":"21898"},{"id":"21934"}],"tools":[{"id":"21888"},{"id":"21889"},{"id":"21890"},{"id":"21891"},{"id":"21892"},{"id":"21893"},{"id":"21894"},{"id":"21895"},{"id":"21924"},{"id":"21925"},{"id":"21926"},{"id":"21927"},{"id":"21928"},{"id":"21929"},{"id":"21930"},{"id":"21931"}]},"id":"21996","type":"ProxyToolbar"},{"attributes":{},"id":"21974","type":"BasicTickFormatter"},{"attributes":{},"id":"21927","type":"WheelZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21896","type":"BoxAnnotation"},{"attributes":{},"id":"21992","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21933","type":"PolyAnnotation"},{"attributes":{},"id":"21993","type":"Selection"},{"attributes":{},"id":"21888","type":"ResetTool"},{"attributes":{},"id":"21978","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21897","type":"PolyAnnotation"},{"attributes":{},"id":"21979","type":"Selection"},{"attributes":{"children":[[{"id":"21871"},0,0],[{"id":"21907"},0,1]]},"id":"21995","type":"GridBox"},{"attributes":{"axis":{"id":"21884"},"dimension":1,"ticker":null},"id":"21887","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21979"},"selection_policy":{"id":"21978"}},"id":"21943","type":"ColumnDataSource"},{"attributes":{},"id":"21980","type":"UnionRenderers"},{"attributes":{},"id":"21981","type":"Selection"},{"attributes":{},"id":"21872","type":"DataRange1d"},{"attributes":{},"id":"21876","type":"LinearScale"},{"attributes":{"children":[{"id":"21997"},{"id":"21995"}]},"id":"21998","type":"Column"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21945","type":"Circle"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21944","type":"Circle"},{"attributes":{"data_source":{"id":"21943"},"glyph":{"id":"21944"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21945"},"selection_glyph":null,"view":{"id":"21947"}},"id":"21946","type":"GlyphRenderer"},{"attributes":{},"id":"21878","type":"LinearScale"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21981"},"selection_policy":{"id":"21980"}},"id":"21952","type":"ColumnDataSource"},{"attributes":{"source":{"id":"21952"}},"id":"21954","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21888"},{"id":"21889"},{"id":"21890"},{"id":"21891"},{"id":"21892"},{"id":"21893"},{"id":"21894"},{"id":"21895"}]},"id":"21898","type":"Toolbar"},{"attributes":{"source":{"id":"21943"}},"id":"21947","type":"CDSView"},{"attributes":{"below":[{"id":"21880"}],"center":[{"id":"21883"},{"id":"21887"}],"left":[{"id":"21884"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21946"},{"id":"21948"},{"id":"21949"},{"id":"21950"},{"id":"21953"}],"title":{"id":"21955"},"toolbar":{"id":"21898"},"toolbar_location":null,"x_range":{"id":"21872"},"x_scale":{"id":"21876"},"y_range":{"id":"21874"},"y_scale":{"id":"21878"}},"id":"21871","subtype":"Figure","type":"Plot"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21951","type":"Dash"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21972"},"ticker":{"id":"21885"}},"id":"21884","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21948","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21949","type":"Span"},{"attributes":{"data_source":{"id":"21966"},"glyph":{"id":"21965"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21968"}},"id":"21967","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"21952"},"glyph":{"id":"21951"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21954"}},"id":"21953","type":"GlyphRenderer"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21986"},"ticker":{"id":"21917"}},"id":"21916","type":"LinearAxis"},{"attributes":{"below":[{"id":"21916"}],"center":[{"id":"21919"},{"id":"21923"}],"left":[{"id":"21920"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21960"},{"id":"21962"},{"id":"21963"},{"id":"21964"},{"id":"21967"}],"title":{"id":"21969"},"toolbar":{"id":"21934"},"toolbar_location":null,"x_range":{"id":"21908"},"x_scale":{"id":"21912"},"y_range":{"id":"21910"},"y_scale":{"id":"21914"}},"id":"21907","subtype":"Figure","type":"Plot"},{"attributes":{"text":"tau"},"id":"21955","type":"Title"},{"attributes":{"end":1,"start":-0.05},"id":"21910","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21984"},"ticker":{"id":"21921"}},"id":"21920","type":"LinearAxis"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21965","type":"Dash"},{"attributes":{},"id":"21912","type":"LinearScale"},{"attributes":{},"id":"21881","type":"BasicTicker"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21950","type":"Span"},{"attributes":{},"id":"21914","type":"LinearScale"},{"attributes":{"overlay":{"id":"21933"}},"id":"21928","type":"LassoSelectTool"},{"attributes":{"end":1,"start":-0.05},"id":"21874","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21959","type":"Circle"},{"attributes":{},"id":"21917","type":"BasicTicker"},{"attributes":{"axis":{"id":"21916"},"ticker":null},"id":"21919","type":"Grid"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21958","type":"Circle"},{"attributes":{"axis":{"id":"21880"},"ticker":null},"id":"21883","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21991"},"selection_policy":{"id":"21990"}},"id":"21957","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"21920"},"dimension":1,"ticker":null},"id":"21923","type":"Grid"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21993"},"selection_policy":{"id":"21992"}},"id":"21966","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"21957"},"glyph":{"id":"21958"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21959"},"selection_glyph":null,"view":{"id":"21961"}},"id":"21960","type":"GlyphRenderer"},{"attributes":{},"id":"21921","type":"BasicTicker"},{"attributes":{},"id":"21924","type":"ResetTool"},{"attributes":{},"id":"21885","type":"BasicTicker"},{"attributes":{"source":{"id":"21957"}},"id":"21961","type":"CDSView"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21964","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21962","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21963","type":"Span"}],"root_ids":["21998"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"e52fe6cb-5b67-4c44-a799-346f6c42eeb7","root_ids":["21998"],"roots":{"21998":"8d7d6506-0758-4310-9147-5a22a12f630b"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();