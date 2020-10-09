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
    
      
      
    
      var element = document.getElementById("30e51ae7-ec51-4404-a9d2-3d3ca7ceb477");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '30e51ae7-ec51-4404-a9d2-3d3ca7ceb477' but no matching script tag was found.")
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
                    
                  var docs_json = '{"fe0d7fba-ed12-4c7a-b2f9-24116e80ab60":{"roots":{"references":[{"attributes":{"callback":null},"id":"22004","type":"HoverTool"},{"attributes":{"source":{"id":"22025"}},"id":"22027","type":"CDSView"},{"attributes":{"text":"mu"},"id":"22042","type":"Title"},{"attributes":{"overlay":{"id":"21969"}},"id":"21963","type":"BoxZoomTool"},{"attributes":{"toolbars":[{"id":"21971"},{"id":"22007"}],"tools":[{"id":"21961"},{"id":"21962"},{"id":"21963"},{"id":"21964"},{"id":"21965"},{"id":"21966"},{"id":"21967"},{"id":"21968"},{"id":"21997"},{"id":"21998"},{"id":"21999"},{"id":"22000"},{"id":"22001"},{"id":"22002"},{"id":"22003"},{"id":"22004"}]},"id":"22069","type":"ProxyToolbar"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"22036","type":"Span"},{"attributes":{"data_source":{"id":"22030"},"glyph":{"id":"22031"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"22032"},"selection_glyph":null,"view":{"id":"22034"}},"id":"22033","type":"GlyphRenderer"},{"attributes":{"callback":null},"id":"21968","type":"HoverTool"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"22038","type":"Dash"},{"attributes":{},"id":"21961","type":"ResetTool"},{"attributes":{"data_source":{"id":"22039"},"glyph":{"id":"22038"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"22041"}},"id":"22040","type":"GlyphRenderer"},{"attributes":{},"id":"21967","type":"SaveTool"},{"attributes":{"overlay":{"id":"21970"}},"id":"21965","type":"LassoSelectTool"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"22035","type":"Span"},{"attributes":{},"id":"21966","type":"UndoTool"},{"attributes":{"source":{"id":"22030"}},"id":"22034","type":"CDSView"},{"attributes":{},"id":"21962","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"22063"},"selection_policy":{"id":"22064"}},"id":"22030","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22031","type":"Circle"},{"attributes":{"source":{"id":"22039"}},"id":"22041","type":"CDSView"},{"attributes":{"data_source":{"id":"22025"},"glyph":{"id":"22024"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"22027"}},"id":"22026","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"22021","type":"Span"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"22005","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"22037","type":"Span"},{"attributes":{"text":"tau"},"id":"22028","type":"Title"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"22022","type":"Span"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"22023","type":"Span"},{"attributes":{},"id":"22047","type":"BasicTickFormatter"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"22024","type":"Dash"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"22053"},"selection_policy":{"id":"22054"}},"id":"22025","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"22006","type":"PolyAnnotation"},{"attributes":{"source":{"id":"22016"}},"id":"22020","type":"CDSView"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21969","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"22006"}},"id":"22001","type":"LassoSelectTool"},{"attributes":{},"id":"22063","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21970","type":"PolyAnnotation"},{"attributes":{},"id":"22064","type":"UnionRenderers"},{"attributes":{},"id":"22051","type":"Selection"},{"attributes":{},"id":"22052","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22017","type":"Circle"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"22047"},"ticker":{"id":"21954"}},"id":"21953","type":"LinearAxis"},{"attributes":{},"id":"22065","type":"Selection"},{"attributes":{},"id":"22066","type":"UnionRenderers"},{"attributes":{},"id":"21964","type":"WheelZoomTool"},{"attributes":{},"id":"22053","type":"Selection"},{"attributes":{},"id":"22054","type":"UnionRenderers"},{"attributes":{},"id":"21945","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21997"},{"id":"21998"},{"id":"21999"},{"id":"22000"},{"id":"22001"},{"id":"22002"},{"id":"22003"},{"id":"22004"}]},"id":"22007","type":"Toolbar"},{"attributes":{"axis":{"id":"21957"},"dimension":1,"ticker":null},"id":"21960","type":"Grid"},{"attributes":{"children":[{"id":"22070"},{"id":"22068"}]},"id":"22071","type":"Column"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"22051"},"selection_policy":{"id":"22052"}},"id":"22016","type":"ColumnDataSource"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"22065"},"selection_policy":{"id":"22066"}},"id":"22039","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22032","type":"Circle"},{"attributes":{},"id":"21951","type":"LinearScale"},{"attributes":{},"id":"22002","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21961"},{"id":"21962"},{"id":"21963"},{"id":"21964"},{"id":"21965"},{"id":"21966"},{"id":"21967"},{"id":"21968"}]},"id":"21971","type":"Toolbar"},{"attributes":{"below":[{"id":"21989"}],"center":[{"id":"21992"},{"id":"21996"}],"left":[{"id":"21993"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"22033"},{"id":"22035"},{"id":"22036"},{"id":"22037"},{"id":"22040"}],"title":{"id":"22042"},"toolbar":{"id":"22007"},"toolbar_location":null,"x_range":{"id":"21981"},"x_scale":{"id":"21985"},"y_range":{"id":"21983"},"y_scale":{"id":"21987"}},"id":"21980","subtype":"Figure","type":"Plot"},{"attributes":{"toolbar":{"id":"22069"},"toolbar_location":"above"},"id":"22070","type":"ToolbarBox"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"22045"},"ticker":{"id":"21958"}},"id":"21957","type":"LinearAxis"},{"attributes":{"data_source":{"id":"22016"},"glyph":{"id":"22017"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"22018"},"selection_glyph":null,"view":{"id":"22020"}},"id":"22019","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"21953"}],"center":[{"id":"21956"},{"id":"21960"}],"left":[{"id":"21957"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"22019"},{"id":"22021"},{"id":"22022"},{"id":"22023"},{"id":"22026"}],"title":{"id":"22028"},"toolbar":{"id":"21971"},"toolbar_location":null,"x_range":{"id":"21945"},"x_scale":{"id":"21949"},"y_range":{"id":"21947"},"y_scale":{"id":"21951"}},"id":"21944","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"22059"},"ticker":{"id":"21990"}},"id":"21989","type":"LinearAxis"},{"attributes":{"end":1,"start":-0.05},"id":"21983","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"22057"},"ticker":{"id":"21994"}},"id":"21993","type":"LinearAxis"},{"attributes":{},"id":"21985","type":"LinearScale"},{"attributes":{},"id":"21954","type":"BasicTicker"},{"attributes":{"end":1,"start":-0.05},"id":"21947","type":"DataRange1d"},{"attributes":{},"id":"21987","type":"LinearScale"},{"attributes":{},"id":"22003","type":"SaveTool"},{"attributes":{},"id":"21990","type":"BasicTicker"},{"attributes":{"axis":{"id":"21989"},"ticker":null},"id":"21992","type":"Grid"},{"attributes":{},"id":"21949","type":"LinearScale"},{"attributes":{"axis":{"id":"21953"},"ticker":null},"id":"21956","type":"Grid"},{"attributes":{"children":[[{"id":"21944"},0,0],[{"id":"21980"},0,1]]},"id":"22068","type":"GridBox"},{"attributes":{},"id":"22057","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"21993"},"dimension":1,"ticker":null},"id":"21996","type":"Grid"},{"attributes":{},"id":"21994","type":"BasicTicker"},{"attributes":{},"id":"22000","type":"WheelZoomTool"},{"attributes":{},"id":"21958","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"22018","type":"Circle"},{"attributes":{},"id":"22059","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"22005"}},"id":"21999","type":"BoxZoomTool"},{"attributes":{},"id":"22045","type":"BasicTickFormatter"},{"attributes":{},"id":"21998","type":"PanTool"},{"attributes":{},"id":"21997","type":"ResetTool"},{"attributes":{},"id":"21981","type":"DataRange1d"}],"root_ids":["22071"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"fe0d7fba-ed12-4c7a-b2f9-24116e80ab60","root_ids":["22071"],"roots":{"22071":"30e51ae7-ec51-4404-a9d2-3d3ca7ceb477"}}];
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